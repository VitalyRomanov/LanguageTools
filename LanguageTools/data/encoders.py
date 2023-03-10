import hashlib

from LanguageTools.utils.file import write_mapping_to_json, read_mapping_from_json


class ValueEncoder:
    def __init__(self, *, values=None, default=None, value_to_code=None):
        self._default = default

        self._initialize(values, value_to_code)

    def _initialize(self, values, value_to_code):
        if values is not None:
            values = sorted(values)
            self._value_to_code = dict(zip(values, range(len(values))))
            self._values = values
        else:
            assert value_to_code is not None
            if self._default is not None:
                assert self._default in value_to_code, f"Provided default value is not in mapping: {self._default}"
            self._value_to_code = value_to_code
            self._values = self._get_ordered_values(value_to_code)

    def set_default(self, default):
        self._default = default

    @property
    def default(self):
        return self._default

    def __repr__(self):
        return repr(self._value_to_code)

    def __getitem__(self, item):
        if item in self._value_to_code:
            return self._value_to_code[item]
        else:
            return self._default

    def inverse(self, item):
        return self._values[item]

    def get(self, key, default=0):
        if key in self._value_to_code:
            return self._value_to_code[key]
        else:
            return default

    def __len__(self):
        return len(self._values)

    def save(self, path):
        # write_mapping_to_json(self._value_to_code, path)
        write_mapping_to_json(self.__dict__, path)

    @classmethod
    def _get_ordered_values(cls, value_to_code):
        return list(zip(*sorted([item for item in value_to_code.items()], key=lambda x: x[1])))[0]

    @classmethod
    def load(cls, path):
        internal_dict = read_mapping_from_json(path)
        new = cls(values=[])
        new.__dict__.update(internal_dict)
        return new


class HashingValueEncoder:
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets

    @property
    def default(self):
        return self._num_buckets

    def __getitem__(self, item):
        return int(hashlib.md5(item.encode('utf-8')).hexdigest(), 16) % max((self._num_buckets - 1), 1)

    def __len__(self):
        return self._num_buckets

    def save(self, path):
        write_mapping_to_json(self.__dict__, path)

    @classmethod
    def load(cls, path):
        internal_dict = read_mapping_from_json(path)
        new = cls(0)
        new.__dict__.update(internal_dict)
        return new


class ShiftingValueEncoder:
    def __init__(self):
        pass

    @property
    def default(self):
        return 0

    def __getitem__(self, item):
        if item == "O":
            return self.default
        return item + 1


class TagMap(ValueEncoder):
    def __init__(self, values):
        super().__init__(values=values)


class ValueEmbedder(ValueEncoder):
    def __init__(self, values=None, default=None, value_to_code=None):
        assert value_to_code is not None
        super(ValueEmbedder, self).__init__(
            values=values, default=default, value_to_code=value_to_code
        )

    def _get_ordered_values(self, *args, **kwargs):
        return None

    def inverse(self, *args, **kwargs):
        raise NotImplemented("This API is not available for this class")


class NoMaskEncoder(ValueEncoder):
    def __init__(self, default=False, *args, **kwargs):
        super(NoMaskEncoder, self).__init__(default=default)

    def _initialize(self, values, value_to_code):
        pass

    def __getitem__(self, item):
        if item == "O":
            return False
        else:
            return True

    def get(self, key, default=0):
        return self.__getitem__(key)


class OMaskEncoder(NoMaskEncoder):
    def __init__(self, default=True, *args, **kwargs):
        super(OMaskEncoder, self).__init__(default=default)

    def __getitem__(self, item):
        if item == "O":
            return True
        else:
            return False


def tag_map_from_sentences(sentences):
    """
    Map tags to an integer values
    :param sentences: list of tags for sentences
    :return: mapping from tags to integers and mapping from integers to tags
    """
    tags = set()

    # find unique tags
    for s in sentences:
        if s is None:
            continue
        tags.add(s)

    return TagMap(list(tags))
