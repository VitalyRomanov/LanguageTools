from __future__ import unicode_literals, print_function

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod

from collections import defaultdict
from copy import copy
# from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from time import time
from typing import Union

from tqdm import tqdm

from LanguageTools.utils.configurable import Configurable
from LanguageTools.data.encoders import TagMap
from LanguageTools.models.abstract_model import AbstractModel
from LanguageTools.utils.file import write_mapping_to_json
from nhcli import ArgumentConfigParser


# @dataclass
# class AbstractTrainerSpecification:
#     output_path: str = None
#     weight_decay: float = 0.01
#     learning_rate: float = 0.001
#     learning_rate_decay: float = 0.99
#     finetune: bool = False
#     pretraining_epochs: int = 0
#     epochs: int = 10
#     ckpt_path: str = None
#     gpu: int = -1


class AbstractModelTrainer(Configurable, ABC):
    model = None
    batcher = None
    device = None
    use_cuda = None
    classes_for = None

    _timestamp = None

    # config_specification = AbstractTrainerSpecification

    def __init__(
            self, train_data, test_data, model_params, data_params, *,
            trainer_output_path: str = None, weight_decay: float = 0.01, learning_rate: float = 0.001,
            learning_rate_decay: float = 0.99, finetune: bool = False, pretraining_epochs: int = 0,
            epochs: int = 10, ckpt_path: str = None, gpu: int = -1
    ):
        self.config = {
            "trainer_output_path": trainer_output_path,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "learning_rate_decay": learning_rate_decay,
            "finetune": finetune,
            "pretraining_epochs": pretraining_epochs,
            "epochs": epochs,
            "ckpt_path": ckpt_path,
            "gpu": gpu
        }
        self.model_params = model_params
        self.data_params = data_params

        self.train_data = train_data
        self.test_data = test_data

        self.set_gpu()

    @property
    def trainer_output_path(self):
        assert self.config["trainer_output_path"] is not None, "Specify trainer_output_path"
        return self.config["trainer_output_path"]

    @property
    def weight_decay(self):
        return self.config["weight_decay"]

    @property
    def learning_rate(self):
        return self.config["learning_rate"]

    @property
    def learning_rate_decay(self):
        return self.config["learning_rate_decay"]

    @property
    def data_path(self):
        return self.config["data_path"]

    @property
    def batch_size(self):
        return self.config["batch_size"]

    @property
    def finetune(self):
        return self.config["finetune"]

    @property
    def pretraining_epochs(self):
        return self.config["pretraining_epochs"]

    @property
    def max_seq_len(self):
        return self.data_params["max_seq_len"]

    @property
    def epochs(self):
        return self.config["epochs"]

    @property
    def ckpt_path(self):
        return self.config["ckpt_path"]

    @property
    def gpu_id(self):
        return self.config["gpu"]

    @property
    def best_score_metric(self):
        return "F1"

    @abstractmethod
    def set_gpu(self):
        if self.gpu_id == -1:
            self.use_cuda = False
            self.device = "cpu"
        else:
            self.use_cuda = True
            self.device = f"cuda:{self.gpu_id}"

    @classmethod
    @abstractmethod
    def get_batcher_class(cls) -> Configurable:
        return None

    @classmethod
    @abstractmethod
    def get_model_class(cls) -> Union[Configurable, AbstractModel]:
        return None

    def get_training_dir(self):
        if not hasattr(self, "_timestamp") or self._timestamp is None:
            self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
        return Path(self.trainer_output_path).joinpath(self._timestamp)

    def get_batcher(self, *args, **kwargs):
        batcher_class = self.get_batcher_class()
        return batcher_class(*args, **self.data_params, **kwargs)

    def get_model(self, *args, **kwargs):
        model_class = self.get_model_class()
        model = model_class(*args, **self.model_params, **kwargs)
        if self.ckpt_path is not None:
            ckpt_path = os.path.join(self.ckpt_path, "checkpoint")
            model = self.load_checkpoint(model, ckpt_path)
        return model

    # @abstractmethod
    def get_dataloaders(self, **kwargs):

        if self.ckpt_path is not None:
            tagmap = TagMap.load(Path(self.ckpt_path).joinpath("tagmap.json"))
        else:
            tagmap = None

        train_batcher = self.get_batcher(
            self.train_data, tagmap=tagmap, **kwargs
        )
        test_batcher = self.get_batcher(
            self.test_data, tagmap=train_batcher.tagmap if tagmap is None else tagmap, **kwargs
        )
        return train_batcher, test_batcher

    @abstractmethod
    def save_checkpoint(self, model, path):
        model.save_weights(path)

    # noinspection PyMethodMayBeStatic
    def save_params(self, path, params, **kwargs):
        params = copy(params)
        params.update(kwargs)
        write_mapping_to_json(params, path.joinpath("params.json"))

    @abstractmethod
    def load_checkpoint(self, model, path):
        model = model.load_weights(path)
        return model

    @abstractmethod
    def _create_summary_writer(self, path):
        self.summary_writer = None

    @abstractmethod
    def _write_to_summary(self, label, value, step):
        ...

    @abstractmethod
    def _create_optimizer(self, model):
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def _lr_scheduler_step(self):
        ...

    @classmethod
    def _format_batch(cls, batch, device):
        pass

    @classmethod
    @abstractmethod
    def compute_loss_and_scores(
            cls, model, token_ids, prefix, suffix, graph_ids, labels, lengths, graph_embs=None, extra_mask=None,
            class_weights=None, scorer=None, finetune=False, vocab_mapping=None, training=False
    ):
        scores = {}
        return scores

    @classmethod
    @abstractmethod
    def make_step(
            cls, batch, model, optimizer, train=False, **kwargs
    ):
        scores = {}
        return scores

    @staticmethod
    @abstractmethod
    def set_model_training(model):
        pass

    @staticmethod
    @abstractmethod
    def set_model_evaluation(model):
        pass

    def iterate_batches(self, model, batches, epoch, num_train_batches, train_scores, scorer, train=True):
        scores_for_averaging = defaultdict(list)

        batch_count = 0

        if train is True:
            self.set_model_training(model)
        else:
            self.set_model_evaluation(model)

        # noinspection PyArgumentList
        for ind, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch}")):
            self._format_batch(batch, self.device)
            scores = self.make_step(batch, model, self.optimizer, train=train)

            batch_count += 1

            scores["batch_size"] = batch['tok_ids'].shape[0]
            for score, value in scores.items():
                self._write_to_summary(f"{score}/{'Train' if train else 'Test'}", value,
                                       epoch * num_train_batches + ind)
                scores_for_averaging[score].append(value)
            train_scores.append(scores_for_averaging)

        return num_train_batches

    def iterate_epochs(self, train_batches, test_batches, epochs, model, scorer, save_ckpt_fn):

        num_train_batches = len(train_batches)
        num_test_batches = len(test_batches)
        train_scores = []
        test_scores = []

        train_average_scores = []
        test_average_scores = []

        best_score = 0.

        try:
            for e in range(epochs):

                start = time()

                num_train_batches = self.iterate_batches(
                    model, train_batches, e, num_train_batches, train_scores, scorer, train=True
                )

                num_test_batches = self.iterate_batches(
                    model, test_batches, e, num_test_batches, test_scores, scorer, train=False
                )

                epoch_time = time() - start

                def print_scores(scores, average_scores, partition):
                    for score, value in scores.items():
                        if score == "batch_size":
                            continue
                        avg_value = sum(value) / len(value)
                        average_scores[score] = avg_value
                        print(f"{partition} {score}: {avg_value: .4f}", end=" ")
                        self._write_to_summary(f"Average {score}/{partition}", avg_value, e)

                train_average_scores.append({})
                test_average_scores.append({})

                print(f"\nEpoch: {e}, {epoch_time: .2f} s", end=" ")
                print_scores(train_scores[-1], train_average_scores[-1], "Train")
                print_scores(test_scores[-1], test_average_scores[-1], "Test")
                print("\n")

                if save_ckpt_fn is not None and test_average_scores[-1][self.best_score_metric] > best_score:
                    save_ckpt_fn()
                    best_score = test_average_scores[-1][self.best_score_metric]

                self._lr_scheduler_step()
        except KeyboardInterrupt:
            pass

        return train_scores, test_scores, train_average_scores, test_average_scores

    def train(
            self, model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01,
            learning_rate_decay=1., finetune=False, save_ckpt_fn=None, no_localization=False
    ):

        self._create_optimizer(model)

        with self.summary_writer.as_default():
            train_scores, test_scores, train_average_scores, test_average_scores = self.iterate_epochs(
                train_batches, test_batches, epochs, model, scorer, save_ckpt_fn
            )
        return train_scores, test_scores, train_average_scores, test_average_scores

    # noinspection PyArgumentList
    # @abstractmethod
    def train_model(self):

        train_batcher, test_batcher = self.get_dataloaders()

        trial_dir = self.get_training_dir()
        trial_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Running trial: {str(trial_dir)}")
        self._create_summary_writer(trial_dir)

        # self.save_params(
        #     trial_dir, {
        #         "MODEL_PARAMS": self.model_params,
        #         "TRAINER_PARAMS": self.config,
        #         "model_class": self.model.__class__.__name__,
        #         "batcher_class": self.batcher.__class__.__name__
        #     }
        # )

        model = self.get_model()

        # def save_ckpt_fn():
        #     checkpoint_path = os.path.join(trial_dir, "checkpoint")
        #     self.save_checkpoint(model, checkpoint_path)

        train_scores, test_scores, train_average_scores, test_average_scores = self.train(
            model=model, train_batches=train_batcher, test_batches=test_batcher
        )

        metadata = {
            "train_scores": train_scores,
            "test_scores": test_scores,
            "train_average_scores": train_average_scores,
            "test_average_scores": test_average_scores,
        }

        with open(os.path.join(trial_dir, "train_data.json"), "w") as metadata_sink:
            metadata_sink.write(json.dumps(metadata, indent=4))

        pickle.dump(train_batcher.tagmap, open(os.path.join(trial_dir, "tag_types.pkl"), "wb"))

    @classmethod
    def add_arguments_from_specification(cls, parser, specification):
        # for field, spec in specification.__dataclass_fields__.items():
        for field, (def_, type_) in specification.items():
            kwargs = {
                "default": def_,
                "type": type_
            }
            if type_ == bool or def_ is False:
                kwargs["action"] = "store_true"
                kwargs.pop("type", None)
            parser.add_argument(f"--{field}", **kwargs)

    @classmethod
    def build_config_parser(cls):

        args = ArgumentConfigParser()

        model_group = args.add_argument_group("MODEL")
        cls.add_arguments_from_specification(model_group, cls.get_model_class().get_config_specification())

        training_group = args.add_argument_group("TRAINING")
        cls.add_arguments_from_specification(training_group, cls.get_config_specification())

        data_group = args.add_argument_group("DATA")
        cls.add_arguments_from_specification(data_group, cls.get_batcher_class().get_config_specification())
        data_group.add_argument("--data_path", default=None)

        args.parse()

        return args
