import logging
from copy import copy
from time import time

import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Dropout

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Embedding, concatenate
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from tqdm import tqdm

from LanguageTools.data.batcher import TaggerBatch





class DefaultEmbedding(Layer):
    """
    Creates an embedder that provides the default value for the index -1. The default value is a zero-vector
    """
    def __init__(self, init_vectors=None, shape=None, trainable=True):
        super(DefaultEmbedding, self).__init__()

        if init_vectors is not None:
            self.embs = tf.Variable(init_vectors, dtype=tf.float32,
                           trainable=trainable, name="default_embedder_var")
            shape = init_vectors.shape
        else:
            # TODO
            # the default value is no longer constant. need to replace this with a standard embedder
            self.embs = tf.Variable(tf.random.uniform(shape=(shape[0], shape[1]), dtype=tf.float32),
                               name="default_embedder_pad")
        self.pad = tf.zeros(shape=(1, shape[1]), name="default_embedder_pad")

    def call(self, ids, mask=None, trainable=None):
        emb_matr = tf.concat([
            self.pad,
            self.embs
        ], axis=0)
        return tf.nn.embedding_lookup(params=emb_matr, ids=ids)

    @property
    def num_embeddings(self):
        return self.embs.shape[0]


class TextCnnLayer(Model):
    def __init__(self, out_dim, kernel_shape, activation=None):
        super(TextCnnLayer, self).__init__()

        self.kernel_shape = kernel_shape
        self.out_dim = out_dim

        self.textConv = Conv2D(filters=out_dim, kernel_size=kernel_shape,
                                  activation=activation, data_format='channels_last')

        padding_size = (self.kernel_shape[0] - 1) // 2
        assert padding_size * 2 + 1 == self.kernel_shape[0]
        self.pad_constant = tf.constant([[0, 0], [padding_size, padding_size], [0, 0], [0, 0]])

        self.supports_masking = True

    def call(self, x, training=None, mask=None):
        padded = tf.pad(x, self.pad_constant)
        # emb_sent_exp = tf.expand_dims(input, axis=3)
        convolve = self.textConv(padded)
        return tf.squeeze(convolve, axis=-2)


class TextCnnEncoder(Model):
    """
    TextCnnEncoder model for classifying tokens in a sequence. The model uses following pipeline:

    token_embeddings (provided from outside) ->
    several convolutional layers, get representations for all tokens ->
    pass representation for all tokens through a dense network ->
    classify each token
    """
    def __init__(self,
                 h_sizes, seq_len,
                 cnn_win_size, dense_size, out_dim,
                 activation=None, dense_activation=None, drop_rate=0.2):
        """

        :param input_size: dimensionality of input embeddings
        :param h_sizes: sizes of hidden CNN layers, internal dimensionality of token embeddings
        :param seq_len: maximum sequence length
        :param cnn_win_size: width of cnn window
        :param dense_size: number of unius in dense network
        :param num_classes: number of output units
        :param activation: activation for cnn
        :param dense_activation: activation for dense layers
        :param drop_rate: dropout rate for dense network
        """
        super(TextCnnEncoder, self).__init__()

        self.seq_len = seq_len
        self.h_sizes = h_sizes
        self.dense_size = dense_size
        self.out_dim = out_dim
        self.cnn_win_size = cnn_win_size
        self.activation = activation
        self.dense_activation = dense_activation
        self.drop_rate = drop_rate

        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3

        input_size = input_shape[2]

        def infer_kernel_sizes(h_sizes):
            """
            Compute kernel sizes from the desired dimensionality of hidden layers
            :param h_sizes:
            :return:
            """
            kernel_sizes = copy(h_sizes)
            kernel_sizes.pop(-1) # pop last because it is the output of the last CNN layer
            kernel_sizes.insert(0, input_size) # the first kernel size should be (cnn_win_size, input_size)
            kernel_sizes = [(self.cnn_win_size, ks) for ks in kernel_sizes]
            return kernel_sizes

        kernel_sizes = infer_kernel_sizes(self.h_sizes)

        self.layers_tok = [TextCnnLayer(out_dim=h_size, kernel_shape=kernel_size, activation=self.activation)
            for h_size, kernel_size in zip(self.h_sizes, kernel_sizes)]

        if self.dense_activation is None:
            dense_activation = self.activation

        self.dense_1 = Dense(self.dense_size, activation=self.dense_activation)
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.drop_rate)
        self.dense_2 = Dense(self.out_dim, activation=None) # logits
        self.dropout_2 = tf.keras.layers.Dropout(rate=self.drop_rate)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, embs, training=True, mask=None):

        x = embs # shape (?, seq_len, input_size)

        # pass embeddings through several CNN layers
        for l in self.layers_tok:
            x = l(tf.expand_dims(x, axis=3)) # shape (?, seq_len, h_size)

        x = self.dense_1(x) # shape (? * seq_len, dense_size)
        tag_logits = self.dense_2(x) # shape (? * seq_len, num_classes)

        return tag_logits  # tf.reshape(tag_logits, (-1, seq_len, self.out_dim)) # reshape back, shape (?, seq_len, num_classes)


class FlatDecoder(Layer):
    def __init__(self, out_dims, hidden=100, dropout=0.1):
        super(FlatDecoder, self).__init__()
        self.fc1 = Dense(hidden, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal())
        self.drop = Dropout(rate=dropout)
        self.fc2 = Dense(out_dims)

    def call(self, inputs, training=None, mask=None):
        encoder_out, target = inputs
        return self.fc2(
            self.drop(
                self.fc1(
                    encoder_out,
                    training=training
                ),
                training=training
            ),
            training=training
        ), None


class CNNTagger(Model):
    """
    TypePredictor model predicts types for Python functions using the following inputs
    Tokens: FastText embeddings for tokens trained on a large collection of texts
    Graph: Graph embeddings pretrained with GNN model
    Prefix: Embeddings for the first n characters of a token
    Suffix: Embeddings for the last n characters of a token
    """
    def __init__(
            self, tok_vectors=None, token_embedder_buckets=None, token_embedder_dim=None,
            train_embeddings=True,
            h_sizes=None, dense_size=100, num_classes=None,
            seq_len=100, cnn_win_size=3,
            crf_transitions=None, suffix_prefix_dims=50, suffix_prefix_buckets=1000,

    ):
        """
        Initialize TypePredictor. Model initializes embedding layers and then passes embeddings to TextCnnEncoder model
        :param tok_embedder: Embedder for tokens
        :param train_embeddings: whether to finetune embeddings
        :param h_sizes: hidden layer sizes
        :param dense_size: sizes of dense layers
        :param num_classes: number of output classes
        :param seq_len: maximum length of sentences
        :param cnn_win_size: width of cnn window
        :param crf_transitions: CRF transition probabilities
        :param suffix_prefix_dims: dimensionality of suffix and prefix embeddings
        :param suffix_prefix_buckets: number of suffix and prefix embeddings
        """
        super(CNNTagger, self).__init__()

        if h_sizes is None:
            h_sizes = [500]

        assert num_classes is not None, "set num_classes"

        self.seq_len = seq_len
        self.transition_params = crf_transitions

        # initialize embeddings
        with tf.device('/CPU:0'):
            if tok_vectors is not None:
                self.tok_emb = DefaultEmbedding(init_vectors=tok_vectors, trainable=train_embeddings)
            else:
                assert token_embedder_dim is not None and token_embedder_buckets is not None
                self.tok_emb = DefaultEmbedding(shape=(token_embedder_buckets, token_embedder_dim))
        self.prefix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))
        self.suffix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))

        # compute final embedding size after concatenation
        input_dim = token_embedder_dim + suffix_prefix_dims * 2 #+ graph_embedder.e.shape[1]

        if cnn_win_size % 2 == 0:
            cnn_win_size += 1
            logging.info(f"Window size should be odd. Setting to {cnn_win_size}")

        self.encoder = TextCnnEncoder(
            h_sizes=h_sizes,
            seq_len=seq_len,
            cnn_win_size=cnn_win_size, dense_size=dense_size,
            out_dim=input_dim, activation=tf.nn.relu,
            dense_activation=tf.nn.tanh)

        self.decoder = FlatDecoder(out_dims=num_classes)

        self.crf_transition_params = None

        self.supports_masking = True

    # @tf.function
    def call(self, inputs, training=False, mask=None):
        """
        # Inference
        :param token_ids: ids for tokens, shape (?, seq_len)
        :param prefix_ids: ids for prefixes, shape (?, seq_len)
        :param suffix_ids: ids for suffixes, shape (?, seq_len)
        :param training: whether to finetune embeddings
        :return: logits for token classes, shape (?, seq_len, num_classes)
        """
        token_ids, prefix_ids, suffix_ids = inputs

        assert mask is not None, "Mask is required"

        tok_emb = self.tok_emb(token_ids)
        prefix_emb = self.prefix_emb(prefix_ids)
        suffix_emb = self.suffix_emb(suffix_ids)

        embs = tf.concat([tok_emb,
                          prefix_emb,
                          suffix_emb], axis=-1)

        encoded = self.encoder(embs, training=training, mask=mask)

        logits, _ = self.decoder((encoded, None), training=training, mask=mask) # consider sending input instead of target

        return logits

    def compute_mask(self, inputs, mask=None):
        mask = self.encoder.compute_mask(None, mask=mask)
        return self.decoder.compute_mask(None, mask=mask)


    # @tf.function
    def loss(self, labels, logits, mask=None):
        """
        Compute cross-entropy loss for each meaningful tokens. Mask padded tokens.
        :param logits: shape (?, seq_len, num_classes)
        :param labels: ids of labels, shape (?, seq_len)
        :param lengths: actual sequence lenghts, shape (?,)
        :param class_weights: optionally provide weights for each token, shape (?, seq_len)
        :param extra_mask: mask for hiding some of the token labels, not counting them towards the loss, shape (?, seq_len)
        :return: average cross-entropy loss
        """
        losses = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth=logits.shape[-1]), logits, axis=-1)
        seq_mask = mask # logits._keras_mask# tf.sequence_mask(lengths, self.seq_len)
        loss = tf.reduce_mean(tf.boolean_mask(losses, seq_mask))

        return loss

    def score(self, labels, logits, mask, scorer):
        true_labels = tf.boolean_mask(labels, mask)
        argmax = tf.math.argmax(logits, axis=-1)
        estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)

        p, r, f1 = scorer(true_labels, estimated_labels)

        return p, r, f1

    def train_step(self, batch: TaggerBatch):
        token_ids = batch.tokens
        prefix = batch.prefixes
        suffix = batch.suffixes
        labels = batch.labels
        seq_mask = batch.mask
        lengths = batch.lengths

        with tf.GradientTape() as tape:
            logits = self((token_ids, prefix, suffix), training=True, mask=seq_mask)
            loss = self.compiled_loss(labels, logits)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, tf.argmax(logits, axis=-1))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch: TaggerBatch, scorer=None):
        token_ids = batch.tokens
        prefix = batch.prefixes
        suffix = batch.suffixes
        labels = batch.labels
        seq_mask = batch.mask
        lengths = batch.lengths

        logits = self((token_ids, prefix, suffix), mask=seq_mask)
        loss = self.compiled_loss(labels, logits)

        self.compiled_metrics.update_state(labels, tf.argmax(logits, axis=-1))

        return {m.name: m.result() for m in self.metrics}


    # def train(
    #         self, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01,
    #         learning_rate_decay=1., finetune=False, summary_writer=None, save_ckpt_fn=None
    # ):
    #
    #     # assert summary_writer is not None
    #
    #     lr = tf.Variable(learning_rate, trainable=False)
    #     self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #
    #     train_losses = []
    #     test_losses = []
    #     train_f1s = []
    #     test_f1s = []
    #
    #     num_train_batches = len(train_batches)
    #     num_test_batches = len(test_batches)
    #
    #     best_f1 = 0.
    #
    #     try:
    #
    #         # with summary_writer.as_default():
    #
    #         for e in range(epochs):
    #             losses = []
    #             ps = []
    #             rs = []
    #             f1s = []
    #
    #             start = time()
    #
    #             for ind, batch in enumerate(tqdm(train_batches)):
    #                 # token_ids, graph_ids, labels, class_weights, lengths = b
    #                 loss, p, r, f1 = self.train_step(batch, scorer=scorer)
    #                 losses.append(loss.numpy())
    #                 ps.append(p)
    #                 rs.append(r)
    #                 f1s.append(f1)
    #
    #                 # tf.summary.scalar("Loss/Train", loss, step=e * num_train_batches + ind)
    #                 # tf.summary.scalar("Precision/Train", p, step=e * num_train_batches + ind)
    #                 # tf.summary.scalar("Recall/Train", r, step=e * num_train_batches + ind)
    #                 # tf.summary.scalar("F1/Train", f1, step=e * num_train_batches + ind)
    #
    #             test_alosses = []
    #             test_aps = []
    #             test_ars = []
    #             test_af1s = []
    #
    #             for ind, batch in enumerate(test_batches):
    #                 # token_ids, graph_ids, labels, class_weights, lengths = b
    #                 test_loss, test_p, test_r, test_f1 = self.test_step(batch, scorer=scorer)
    #
    #                 # tf.summary.scalar("Loss/Test", test_loss, step=e * num_test_batches + ind)
    #                 # tf.summary.scalar("Precision/Test", test_p, step=e * num_test_batches + ind)
    #                 # tf.summary.scalar("Recall/Test", test_r, step=e * num_test_batches + ind)
    #                 # tf.summary.scalar("F1/Test", test_f1, step=e * num_test_batches + ind)
    #                 test_alosses.append(test_loss)
    #                 test_aps.append(test_p)
    #                 test_ars.append(test_r)
    #                 test_af1s.append(test_f1)
    #
    #             epoch_time = time() - start
    #
    #             train_losses.append(float(sum(losses) / len(losses)))
    #             train_f1s.append(float(sum(f1s) / len(f1s)))
    #             test_losses.append(float(sum(test_alosses) / len(test_alosses)))
    #             test_f1s.append(float(sum(test_af1s) / len(test_af1s)))
    #
    #             print(
    #                 f"Epoch: {e}, {epoch_time: .2f} s, Train Loss: {train_losses[-1]: .4f}, Train P: {sum(ps) / len(ps): .4f}, Train R: {sum(rs) / len(rs): .4f}, Train F1: {sum(f1s) / len(f1s): .4f}, "
    #                 f"Test loss: {test_losses[-1]: .4f}, Test P: {sum(test_aps) / len(test_aps): .4f}, Test R: {sum(test_ars) / len(test_ars): .4f}, Test F1: {test_f1s[-1]: .4f}")
    #
    #             if save_ckpt_fn is not None and float(test_f1s[-1]) > best_f1:
    #                 save_ckpt_fn()
    #                 best_f1 = float(test_f1s[-1])
    #
    #             lr.assign(lr * learning_rate_decay)
    #
    #     except KeyboardInterrupt:
    #         pass
    #
    #     return train_losses, train_f1s, test_losses, test_f1s


def to_numpy(tensor):
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    else:
        return tf.make_ndarray(tf.make_tensor_proto(tensor))




# @tf.function



def test(model, test_batches, scorer=None):
    test_alosses = []
    test_aps = []
    test_ars = []
    test_af1s = []

    for ind, batch in enumerate(test_batches):
        # token_ids, graph_ids, labels, class_weights, lengths = b
        test_loss, test_p, test_r, test_f1 = test_step(
            model=model, token_ids=batch['tok_ids'],
            prefix=batch['prefix'], suffix=batch['suffix'], graph_ids=batch['graph_ids'],
            labels=batch['tags'], lengths=batch['lens'], extra_mask=batch['hide_mask'],
            # class_weights=batch['class_weights'],
            scorer=scorer
        )

        test_alosses.append(test_loss)
        test_aps.append(test_p)
        test_ars.append(test_r)
        test_af1s.append(test_f1)

    def avg(arr):
        return sum(arr) / len(arr)

    return avg(test_alosses), avg(test_aps), avg(test_ars), avg(test_af1s)

