import inspect
from abc import ABC
from copy import copy

from LanguageTools.utils.file import write_mapping_to_yaml, read_mapping_from_yaml


class Configurable(ABC):
    """
    Configurable allows defining default config for a class. The items that should appear in config should be
    defined as kwonly arguments and should have a default value.

    Usage example:
    class TestConfigurable1(Configurable):
    def __init__(self, arg1: str, arg2, *, arg3 = 1, arg4: str = None):
        ...

    spec = TestConfigurable1.get_config_specification()
    assert spec == {
        "arg3": (1, None),
        "arg4": (None, str)
    }
    """
    config = {}

    def __init__(self, kwargs):
        default_config = self.get_config_specification()
        config = {}
        for var, value in kwargs.items():
            if var in default_config:
                config[var] = value

        assert len(config) == len(default_config), f"Expecting at least {len(default_config)} parameters, but found {len(config)}. Make sure to pass all parameters {default_config}"
        self.config = config

    @classmethod
    def get_error_message(cls):
        return f"Arguments of `Configurable` constructor ({cls.__name__}) should have a default value and type " \
               f"annotation, but the signature is {inspect.signature(cls.__init__)}"

    @classmethod
    def get_config_specification(cls):
        _params = inspect.signature(cls.__init__).parameters

        spec = {}
        for key, val in _params.items():
            # if key in {"self", "cls"}:
            #     continue
            if val.default is inspect._empty:
                continue

            if val.annotation is inspect._empty:
                raise AssertionError(cls.get_error_message())
            spec[key] = (val.default, val.annotation)

        # annotations = cls.__init__.__annotations__
        # defaults = cls.__init__.__kwdefaults__
        # n_config_vars = cls.__init__.__code__.co_kwonlyargcount
        # assert len(defaults) == n_config_vars, cls.get_error_message()
        #
        # spec = {}
        # for key in defaults:
        #     spec[key] = (defaults.get(key, None), annotations.get(key, None))

        return spec

    def save_config(self, path):
        write_mapping_to_yaml(self.config, path)

    def get_config(self):
        return copy(self.config)

    @classmethod
    def load_config(cls, path):
        return read_mapping_from_yaml(path)


