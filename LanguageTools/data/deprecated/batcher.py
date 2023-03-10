import hashlib
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# @dataclass
# class Batch:
#     X: np.ndarray
#     y: np.ndarray
#     lengths: np.ndarray
#     mask: np.ndarray

TaggerBatch = namedtuple("Batch", ["tokens", "prefixes", "suffixes", "labels", "mask", "lengths"])


class TokenHasher:
    def __init__(self, num_buckets):
        self.num_buckets = num_buckets

    @lru_cache(50000)
    def __getitem__(self, token: str):
        return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % self.num_buckets


class Batcher:
    def __init__(
            self, text, labels, batch_size, tokenizer=None, add_outside_label=False,
            word_hasher_bucket_size=500000,
            hasher_bucket_size=2000, wordmap=None, tagmap=None, wordembmap=None
    ):
        self.X = text
        self.y = labels
        self.batch_size = batch_size
        self.set_tokenizer(tokenizer)

        if wordmap is None:
            if wordembmap is not None:
                self.wordmap = defaultdict(lambda: 0) | {key: val + 1 for key, val in wordembmap.items()}
            else:
                # self.wordmap = self.initialize_mapper(default_key="<UNK>", default_value=0)
                self.wordmap = TokenHasher(word_hasher_bucket_size)
        else:
            self.wordmap = wordmap

        if tagmap is None:
            if add_outside_label:
                self.tagmap = self.initialize_mapper(default_key="O", default_value=0)
            else:
                self.tagmap = self.initialize_mapper()
        else:
            self.tagmap = tagmap

        self.prefixmap = TokenHasher(hasher_bucket_size)
        self.suffixmap = TokenHasher(hasher_bucket_size)

        for x_, y_ in zip(self.X, self.y):
            self.add_missing(x_, y_)

    def initialize_mapper(self, default_key=None, default_value=None):
        mapper = defaultdict(lambda: default_value)

        if default_key is not None:
            mapper[default_key] = default_value
        return mapper

    def get_default_tag(self):
        return self.tagmap["O"]

    def set_tokenizer(self, tokenizer):
        if tokenizer is None:
            self.tokenizer = None

    def tokenize(self, text, labels):
        return self.tokenizer(text, labels)

    def encode(self, seq, mapper):
        return list(map(lambda x: mapper[x], seq))

    def get_tokenized(self, text, labels_):
        if self.tokenizer is not None:
            tokens, labels = self.tokenize(text, labels_)
        else:
            tokens, labels = text, labels_
        return tokens, labels

    def add_to_map(self, mapper, values):
        for value in values:
            if value not in mapper:
                mapper[value] = len(mapper)

    def add_missing(self, text, labels):
        tokens, labels = self.get_tokenized(text, labels)

        # self.add_to_map(self.wordmap, tokens)
        self.add_to_map(self.tagmap, labels)

    def prepare_for_batch(self, text, labels_, suff_pref_len=3):
        if self.tokenizer is not None:
            tokens, labels = self.tokenize(text, labels_)
        else:
            tokens, labels = text, labels_

        prefixes = [token[:suff_pref_len] for token in tokens]
        suffixes = [token[-suff_pref_len:] for token in tokens]

        enc_tokens = self.encode(tokens, self.wordmap)
        enc_labels = self.encode(labels, self.tagmap)
        enc_prefixes = self.encode(prefixes, self.prefixmap)
        enc_suffixes = self.encode(suffixes, self.suffixmap)

        return enc_labels, enc_tokens, enc_prefixes, enc_suffixes,

    def pad(self, seq, max_len):
        pad_len = max_len - len(seq)
        return seq + [0] * pad_len

    # @lru_cache
    def form_batch(self, ind):
        lenghts = [len(sent) for sent in self.batch_tokens]
        max_len = max(lenghts)

        padded_tokens = [self.pad(sent, max_len) for sent in self.batch_tokens]
        padded_labels = [self.pad(sent, max_len) for sent in self.batch_labels]
        padded_prefixes = [self.pad(sent, max_len) for sent in self.batch_suffixes]
        padded_suffixes = [self.pad(sent, max_len) for sent in self.batch_prefixes]
        mask = [[True] * len_ + [False] * (max_len - len_) for len_ in lenghts]

        # batch = Batch(
        #     np.array(padded_tokens, dtype=np.int32),
        #     np.array(padded_labels, dtype=np.int32),
        #     np.array(lenghts, dtype=np.int32),
        #     np.array(mask, dtype=np.bool)
        # )

        self.batch_tokens.clear()
        self.batch_labels.clear()
        self.batch_suffixes.clear()
        self.batch_prefixes.clear()

        return TaggerBatch(padded_tokens, padded_prefixes, padded_suffixes, padded_labels, mask, lenghts)

    @property
    def current_batch_size(self):
        return len(self.batch_labels)

    def add_to_current_batch(self, enc_labels, enc_tokens, enc_prefixes=None, enc_suffixes=None):
        self.batch_labels.append(enc_labels)
        self.batch_tokens.append(enc_tokens)
        if enc_prefixes is not None:
            self.batch_prefixes.append(enc_prefixes)
        if enc_suffixes is not None:
            self.batch_suffixes.append(enc_suffixes)

    def batches(self):
        self.batch_labels = []
        self.batch_tokens = []
        self.batch_prefixes = []
        self.batch_suffixes = []
        for ind, (X, y) in enumerate(zip(self.X, self.y)):
            self.add_to_current_batch(*self.prepare_for_batch(X, y))
            if self.current_batch_size == self.batch_size:
                yield self.form_batch(ind)
        if self.current_batch_size > 0:
            yield self.form_batch(ind + 1)

    def __iter__(self):
        return self.batches()

    def __len__(self):
        return len(self.X) // self.batch_size + 1

    @property
    def num_classes(self):
        return len(self.tagmap)