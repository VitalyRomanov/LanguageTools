from pathlib import Path

from LanguageTools.data.batcher import Batcher
from LanguageTools.models.torch_models.entity import EntityAnnotator
from LanguageTools.trainers.TorchModelTrainer import TorchModelTrainer
from LanguageTools.utils.file import read_jsonl


class BertChunkerTrainer(TorchModelTrainer):
    def __init__(
            self, train_data, test_data, model_params, data_params, *,
            trainer_output_path: str = None, weight_decay: float = 0.01, learning_rate: float = 0.001,
            learning_rate_decay: float = 0.99, finetune: bool = False, pretraining_epochs: int = 0,
            epochs: int = 10, ckpt_path: str = None, gpu: int = -1
    ):
        super(BertChunkerTrainer, self).__init__(
            train_data, test_data, model_params, data_params,
            trainer_output_path=trainer_output_path, weight_decay=weight_decay, learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay, finetune=finetune,
            pretraining_epochs=pretraining_epochs, epochs=epochs, ckpt_path=ckpt_path, gpu=gpu
        )

    @classmethod
    def get_batcher_class(cls):
        return Batcher

    @classmethod
    def get_model_class(cls):
        return EntityAnnotator

    def get_model(self, *args, **kwargs):
        return EntityAnnotator(**self.model_params)

    def get_batcher(self, *args, **kwargs):
        batcher_class = self.get_batcher_class()

        from LanguageTools.tokenizers import Tokenizer
        tokenizer = Tokenizer(self.data_params["tokenizer"])
        vocab = tokenizer.get_vocab()

        return batcher_class(*args, wordmap=vocab, **self.data_params, **kwargs)

    @classmethod
    def compute_loss_and_scores(cls, model, token_ids, prefix, suffix, graph_ids, labels, lengths, graph_embs=None,
                                extra_mask=None, class_weights=None, scorer=None, finetune=False, vocab_mapping=None,
                                training=False):
        pass

    @classmethod
    def make_step(cls, batch, model, optimizer, train=False, **kwargs):
        pass


if __name__ == "__main__":

    args = BertChunkerTrainer.build_config_parser()
    config = args.get_config()
    data_path = Path(config["DATA"]["data_path"])
    train_path = data_path.joinpath("train_data.json")
    test_path = data_path.joinpath("test_data.json")

    train_data = read_jsonl(train_path, ents_field="chunks")
    test_data = read_jsonl(test_path, ents_field="chunks")

    model_trainer = BertChunkerTrainer(
        train_data, test_data, model_params=config["MODEL"],
        data_params=config["DATA"], **config["TRAINING"]
    )

    model_trainer.train_model()

    print()
    # read_jsonl