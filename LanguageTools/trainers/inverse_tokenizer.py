from LanguageTools.data.deprecated.batcher import Batcher, TaggerBatch
from LanguageTools.models.CNNModel import CNNTagger

import tensorflow as tf

from LanguageTools.models.embeddings import load_w2v


def compute_precision_recall_f1(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


class F1Metric(tf.keras.metrics.Metric):
    def __init__(self, class_id):
        super().__init__()
        self.p = tf.keras.metrics.Precision(class_id=class_id)
        self.r = tf.keras.metrics.Recall(class_id=class_id)

    @property
    def name(self):
        return "f1_score"

    def update_state(self, labels, predictions, *args, **kwargs):
        self.p.update_state(labels, predictions)
        self.r.update_state(labels, predictions)

    def reset_state(self):
        self.p.reset_state()
        self.r.reset_state()

    def result(self):
        p = self.p.result()
        r = self.r.result()
        return 2 * p * r / (p + r)


class InverseTokenizerTrainer:
    def __init__(self, train_data, test_data, batch_size=64, emb_path=None):
        self.train_text, self.train_spaces = train_data
        self.test_text, self.test_spaces = test_data
        self.batch_size=64
        self.emb_path = emb_path

        self.load_embeddings()
        self.create_batchers()
        self.create_model()
        self.create_optimizer()

    def create_batchers(self):
        self.train_batcher = Batcher(
            text=self.train_text, labels=self.train_spaces, batch_size=self.batch_size,
            wordembmap=self.word_embs.mapping
        )
        self.test_batcher = Batcher(
            text=self.test_text, labels=self.test_spaces, batch_size=self.batch_size,
            tagmap=self.train_batcher.tagmap
        )

    def load_embeddings(self):
        if self.emb_path is not None:
            self.word_embs = load_w2v(self.emb_path)

    def create_model(self):
        self.model = CNNTagger(
            tok_vectors=self.word_embs.vectors,
            # token_embedder_buckets=500000, token_embedder_dim=10,
            h_sizes=[20,20,20], dense_size=20, num_classes=self.train_batcher.num_classes,
            seq_len=100, cnn_win_size=5,
            suffix_prefix_dims=10, suffix_prefix_buckets=2000
        )

    def create_optimizer(self):
        pass

    def create_mask(self, lengths):
        pass

    def train(self):

        # def score(true, predicted, default_tag):
        #     positive_mask = true != default_tag
        #     positive_true, positive_pred = true[positive_mask], predicted[positive_mask]
        #     tp = tf.reduce_sum(tf.cast(positive_true == positive_pred, tf.float32))
        #     fn = len(positive_true) - tp
        #     negative_pred = predicted[~positive_mask]
        #     fp = tf.reduce_sum(tf.cast(negative_pred != default_tag, tf.float32))
        #     return compute_precision_recall_f1(tp, fp, fn)
        #
        # self.model.train(
        #     train_batches=self.train_batcher, test_batches=self.test_batcher, epochs=3,
        #     scorer=lambda true, pred: score(true, pred, default_tag=self.train_batcher.tagmap[True])
        # )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                keras.metrics.Accuracy(),
                keras.metrics.Precision(class_id=self.train_batcher.tagmap[False]),
                keras.metrics.Recall(class_id=self.train_batcher.tagmap[False]),
                F1Metric(class_id=self.train_batcher.tagmap[False])
            ]
        )
        # self.model.run_eagerly = True
        train_data = tf.data.Dataset.from_generator(
            self.train_batcher.batches,
            output_signature=TaggerBatch(
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None, ), dtype=tf.int32),
            )
        ).repeat()

        validation_data = tf.data.Dataset.from_generator(
            self.test_batcher.batches,
            output_signature=TaggerBatch(
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            )
        ).repeat()
        self.model.fit(
            train_data, epochs=30,
            steps_per_epoch=len(self.train_batcher),
            validation_data=validation_data,
            validation_steps=len(self.test_batcher)
        )
        # for batch in train_data:
        #     print(len(batch))
        #     # mask = self.create_mask(batch.lengths)
        #     # self.model(batch.X, batch.y, mask)


if __name__ == "__main__":
    from LanguageTools.data.conll import get_conll_spaces, read_conll
    import sys

    train_data = get_conll_spaces(read_conll(sys.argv[1]))
    test_data = get_conll_spaces(read_conll(sys.argv[2]))
    trainer = InverseTokenizerTrainer(train_data, test_data, emb_path=sys.argv[3])
    trainer.train()