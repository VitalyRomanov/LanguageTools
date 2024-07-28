from abc import ABC

try:
    import torch
    from torch import nn
except ImportError:
    raise ImportError("Install torch: pip install torch")

from LanguageTools.models.abstract_model import AbstractModel


class AbstractTorchModel(AbstractModel, nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        AbstractModel.__init__(self)

    def load_params(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def save(self, path):
        torch.save(self.state_dict(), path)
