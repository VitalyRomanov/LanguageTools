from abc import ABC, abstractmethod
from pathlib import Path

from LanguageTools.utils.configurable import Configurable


class AbstractModel(Configurable, ABC):
    config = {}

    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def loss(self, logits, labels, mask):
        ...

    @abstractmethod
    def score(self, logits, labels, mask):
        ...

    @abstractmethod
    def load_params(self, path):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        path = Path(path)
        config = cls.load_config(path)
        model = cls(**config)
        model.load_params(path)
        return model
