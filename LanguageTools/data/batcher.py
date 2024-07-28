import hashlib
import json
import random
import tempfile
from collections import defaultdict
from copy import copy, deepcopy
from pathlib import Path
from math import ceil
from typing import Dict, Optional, Union, Callable

import numpy as np

from tqdm import tqdm

from LanguageTools.utils.configurable import Configurable
from LanguageTools.data.encoders import TagMap, ValueEncoder
from LanguageTools.tokenizers import biluo_tags_from_offsets as biluo_tags_from_offsets_, Tokenizer
from LanguageTools.utils.file import read_mapping_from_json, write_mapping_to_json

from nhkv import KVStore, get_or_create_storage


def print_token_tag(doc, tags):
    for t, tag in zip(doc, tags):
        print(t, "\t", tag)


def biluo_tags_from_offsets(doc, ents, no_localization):
    ent_tags = biluo_tags_from_offsets_(doc, ents)

    if no_localization:
        tags = []
        for ent in ent_tags:
            parts = ent.split("-")

            assert len(parts) <= 2

            if len(parts) == 2:
                if parts[0] == "B" or parts[0] == "U":
                    tags.append(parts[1])
                else:
                    tags.append("O")
            else:
                tags.append("O")

        ent_tags = tags

    return ent_tags


def fix_incorrect_tags(tags):
    while "-" in tags:
        tags[tags.index("-")] = "O"


class SampleEntry(object):
    # noinspection PyShadowingBuiltins
    def __init__(self, id, text, labels=None, category=None, **kwargs):
        self._storage = dict()
        self._storage["id"] = id
        self._storage["text"] = text
        self._storage["labels"] = labels
        self._storage["category"] = category
        self._storage.update(kwargs)

    def __getattr__(self, item):
        storage = object.__getattribute__(self, "_storage")
        if item in storage:
            return storage[item]
        return super().__getattribute__(item)

    def __repr__(self):
        return repr(self._storage)

    def __getitem__(self, item):
        return self._storage[item]

    def __contains__(self, item):
        return item in self._storage

    def keys(self):
        return list(self._storage.keys())


class MapperSpec:
    def __init__(
            self, field, target_field, encoder, dtype=np.int32, preproc_fn=None,
            encoder_fn: Union[str, Callable] = "seq"
    ):
        self.field = field
        self.target_field = target_field
        self.encoder = encoder
        self.preproc_fn = preproc_fn
        self.dtype = dtype
        self.encoder_fn = encoder_fn


# @dataclass
# class BatcherSpecification:
#     batch_size: int = 32
#     max_seq_len: int = 512
#     sort_by_length: bool = True
#     tokenizer: str = None
#     no_localization: bool = False
#     cache_dir: str = None


class Batcher(Configurable):
    # config_specification = BatcherSpecification

    def __init__(
            self, data, wordmap: Dict[str, int], tagmap: Optional[TagMap] = None, labelmap: Optional[TagMap] = None, *,
            batch_size: int = 32, max_seq_len: int = 512, sort_by_length: bool = True,
            tokenizer: str = None, no_localization: bool = False,
            cache_dir: Path = None, **kwargs
    ):
        super(Batcher, self).__init__(locals())
        # self.config = {
        #     "batch_size": batch_size,
        #     "max_seq_len": max_seq_len,
        #     "sort_by_length": sort_by_length,
        #     "tokenizer": tokenizer,
        #     "no_localization": no_localization,
        #     "cache_dir": cache_dir
        # }

        self._data = data
        self._class_weights = None
        self._nlp = Tokenizer(self._tokenizer)
        self._valid_sentences = 0
        self._filtered_sentences = 0
        self.wordmap = wordmap
        self.tagmap = tagmap
        self.labelmap = labelmap
        self._data_ids = set()
        self._batch_generator = None

        self._create_cache()
        self._prepare_data()
        self._create_mappers(**kwargs)

    @property
    def _batch_size(self):
        return self.config["batch_size"]

    @property
    def _max_seq_len(self):
        return self.config["max_seq_len"]

    @property
    def _tokenizer(self):
        return self.config["tokenizer"]

    @property
    def _no_localization(self):
        return self.config["no_localization"]

    @property
    def _cache_dir(self):
        return self.config["cache_dir"]

    @property
    def _sort_by_length(self):
        return self.config["sort_by_length"]

    @property
    def _data_cache_path(self):
        return self._get_cache_location_name("DataCache")

    # @property
    # def _sent_cache_path(self):
    #     return self._get_cache_location_name("SentCache")

    @property
    def _batch_cache_path(self):
        return self._get_cache_location_name("BatchCache")

    @property
    def _length_cache_path(self):
        return self._get_cache_location_name("LengthCache")

    @property
    def _tagmap_path(self):
        return self._current_cache_dir.joinpath("tagmap.json")

    @property
    def _labelmap_path(self):
        return self._current_cache_dir.joinpath("labelmap.json")

    @property
    def _unique_tags_and_labels_path(self):
        return self._current_cache_dir.joinpath("unique_tags_and_labels.json")

    @property
    def _tag_fields(self):
        return ["tags"]

    @property
    def _category_fields(self):
        return ["category"]

    def num_classes(self, how):
        if how == "tags":
            if len(self.tagmap) > 0:
                return len(self.tagmap)
            else:
                raise Exception("There were no tags in the data")
        elif how == "labels":
            if self.labelmap is not None and len(self.labelmap) > 0:
                return len(self.labelmap)
            else:
                raise Exception("There were no labels in the data")
        else:
            raise ValueError(f"Unrecognized category for classes: {how}")

    def _get_version_code(self):
        signature_dict = {"tokenizer": self._tokenizer, "class_weights": self._class_weights,
                          "_no_localization": self._no_localization, "class": self.__class__.__name__,
                          "wordmap": sorted(self.wordmap.items(), key=lambda x: x[1])}
        if hasattr(self, "_extra_signature_parameters"):
            if hasattr(self, "_extra_signature_parameters_ignore_list"):
                signature_dict.update(
                    {
                        key: val for key, val in self._extra_signature_parameters.items()
                        if key not in self._extra_signature_parameters_ignore_list
                    }
                )
            else:
                signature_dict.update(self._extra_signature_parameters)
        defining_parameters = json.dumps(signature_dict)
        return self._compute_text_id(defining_parameters)

    def _get_cache_location_name(self, cache_name):
        self._check_cache_dir()
        return str(self._current_cache_dir.joinpath(cache_name))

    def _check_cache_dir(self):
        if not hasattr(self, "_current_cache_dir") or self._current_cache_dir is None:
            raise Exception("Cache directory location has not been specified yet")

    def _create_cache(self):
        self._current_cache_dir = self._cache_dir
        if self._current_cache_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            self._current_cache_dir = Path(self._tmp_dir.name)

        self._current_cache_dir = self._current_cache_dir.joinpath(
            f"{self.__class__.__name__}{self._get_version_code()}"
        )
        self._current_cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_cache = get_or_create_storage(KVStore, path=self._data_cache_path)
        self._length_cache = get_or_create_storage(KVStore, path=self._length_cache_path)
        self._batch_cache = get_or_create_storage(KVStore, path=self._batch_cache_path)

    @staticmethod
    def _compute_text_id(text):
        return abs(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)) % 1152921504606846976

    def _prepare_record(self, id_, text, annotations):
        extra = copy(annotations)
        labels = extra.pop("entities", [])  # remove from copy
        extra.update(self._prepare_tokenized_sent((text, annotations)))
        entry = SampleEntry(id=id_, text=text, labels=labels, **extra)
        return entry

    def _update_unique_tags_and_labels(self, unique_tags_and_labels):
        if self._unique_tags_and_labels_path.is_file():
            existing = read_mapping_from_json(self._unique_tags_and_labels_path)
            for field in unique_tags_and_labels:
                if field in existing:
                    existing[field] = set(existing[field])
                    existing[field].update(unique_tags_and_labels[field])
                else:
                    existing[field] = unique_tags_and_labels[field]

            unique_tags_and_labels = existing

        for field in unique_tags_and_labels:
            unique_tags_and_labels[field] = list(unique_tags_and_labels[field])

        write_mapping_to_json(unique_tags_and_labels, self._unique_tags_and_labels_path)

    def _prepare_data(self):
        data_edited = False
        length_edited = True

        unique_tags_and_labels = defaultdict(set)

        def iterate_tags(record, field):
            for label in record[field]:
                yield label

        for text, annotations in tqdm(self._data, desc="Scanning data"):
            id_ = self._compute_text_id(text)
            self._data_ids.add(id_)
            # if id_ not in self._length_cache:
            if self._length_cache.get(id_, None) is None:
                length_edited = True
                # if id_ not in self._data_cache:
                if self._data_cache.get(id_, None) is None:
                    data_edited = True
                    self._data_cache[id_] = (text, annotations)
                entry = self._prepare_record(id_, text, annotations)

                for tag_field in self._tag_fields:
                    unique_tags_and_labels[tag_field].update(set(iterate_tags(entry, tag_field)))

                for cat_field in self._category_fields:
                    cat = entry[cat_field]
                    if cat is not None:
                        unique_tags_and_labels[cat_field].add(entry.category)

                self._length_cache[id_] = len(entry.tokens)

        self._update_unique_tags_and_labels(unique_tags_and_labels)

        if data_edited:
            self._data_cache.save()
        if length_edited:
            self._length_cache.save()

    def _prepare_tokenized_sent(self, sent):
        text, annotations = sent

        doc = self._nlp(text)
        ents = annotations['ents']

        tokens = doc
        # try:
        #     tokens = [t.text for t in tokens]
        # except AttributeError:
        #     pass

        ents_tags = self._biluo_tags_from_offsets(doc, ents, check_localization_parameter=True)
        assert len(tokens) == len(ents_tags)

        output = {
            "tokens": tokens,
            "tags": ents_tags
        }

        output.update(self._parse_additional_tags(text, annotations, doc, output))

        return output

    # def _get_adjustment(self, doc, offsets):
    #     if hasattr(doc, "requires_offset_adjustment") and doc.requires_offset_adjustment:
    #         adjusted_offsets = doc.adjust_offsets(offsets)
    #         tokens_for_biluo_alignment = doc.get_tokens_for_alignment()
    #     else:
    #         adjusted_offsets = offsets
    #         tokens_for_biluo_alignment = doc
    #     return tokens_for_biluo_alignment, adjusted_offsets
    #     # if hasattr(doc, "tokens_for_biluo_alignment"):
    #     #     entity_adjustment_amount = doc.adjustment_amount
    #     #     tokens_for_biluo_alignment = doc.tokens_for_biluo_alignment
    #     # else:
    #     #     entity_adjustment_amount = 0
    #     #     tokens_for_biluo_alignment = doc
    #     # return entity_adjustment_amount, tokens_for_biluo_alignment

    def _biluo_tags_from_offsets(self, doc, tags, check_localization_parameter=False):
        if check_localization_parameter is True:
            no_localization = self._no_localization
        else:
            no_localization = False
        # tokens_for_biluo_alignment, adjusted_offsets = self._get_adjustment(doc, tags)
        ents_tags = biluo_tags_from_offsets(doc, tags, no_localization)
        fix_incorrect_tags(ents_tags)
        return ents_tags

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _parse_additional_tags(self, text, annotations, doc, parsed):
        # for future use
        return {}

    def get_record_with_id(self, id_):
        record = self._data_cache.get(id_, None)
        # if id_ not in self._data_cache:
        if record is None:
            raise KeyError("Record with such id is not found")
        # text, annotations = self._data_cache[id_]
        text, annotations = record
        return self._prepare_record(id_, text, annotations)
        # return self._data_cache[id]

    def _iterate_record_ids(self):
        return list(self._data_ids)

    def _iterate_sorted_by_length(self, limit_max_length=False):
        ids = list(self._iterate_record_ids())
        ids_length = list(zip(ids, list(map(lambda x: self._length_cache[x], ids))))
        for id_, length in sorted(ids_length, key=lambda x: x[1]):
            if id_ not in self._data_ids or limit_max_length and length >= self._max_seq_len:
                continue
            yield id_
            # text, annotations = self._get_record_with_id(id_)
            # yield self._prepare_record(id_, text, annotations)

    def _iterate_records(self, limit_max_length=False, shuffle=False):
        ids = self._iterate_record_ids()
        if shuffle:
            random.shuffle(ids)
        for id_ in ids:
            if limit_max_length and self._length_cache[id_] >= self._max_seq_len:
                continue
            yield id_
            # text, annotations = self._get_record_with_id(id_)
            # yield self._prepare_record(id_, text, annotations)

    def _collect_pad_values_for_fields(self):
        self._default_padding = {}
        for mapper in self._mappers:
            enc = mapper.encoder
            if hasattr(enc, "default"):
                self._default_padding[mapper.target_field] = enc.default

    def get_default_padding(self):
        return deepcopy(self._default_padding)

    # noinspection PyUnusedLocal
    def _create_mappers(self, **kwargs):
        self._mappers = []
        self._create_wordmap_encoder()
        self._create_tagmap_encoder()
        self._create_category_encoder()
        self._create_additional_encoders()

        self._collect_pad_values_for_fields()

    def _create_additional_encoders(self):
        pass

    # noinspection PyUnusedLocal
    def _create_category_encoder(self, **kwargs):
        if self.labelmap is None:
            if self._labelmap_path.is_file():
                labelmap = TagMap.load(self._labelmap_path)
            else:
                if not self._unique_tags_and_labels_path.is_file():
                    return
                unique_tags_and_labels = read_mapping_from_json(self._unique_tags_and_labels_path)
                if "category" not in unique_tags_and_labels:
                    return
                labelmap = TagMap(
                    unique_tags_and_labels["category"]
                )
                labelmap.save(self._labelmap_path)

            self.labelmap = labelmap

        self._mappers.append(
            MapperSpec(field="category", target_field="label", encoder=self.labelmap, encoder_fn="item")
        )

    def _create_tagmap_encoder(self):
        if self.tagmap is None:
            if self._tagmap_path.is_file():
                tagmap = TagMap.load(self._tagmap_path)
            else:
                unique_tags_and_labels = read_mapping_from_json(self._unique_tags_and_labels_path)
                if "tags" not in unique_tags_and_labels:
                    return
                tagmap = TagMap(
                    unique_tags_and_labels["tags"]
                )
                tagmap.set_default(tagmap._value_to_code["O"])
                tagmap.save(self._tagmap_path)

            self.tagmap = tagmap

        self._mappers.append(
            MapperSpec(field="tags", target_field="tags", encoder=self.tagmap)
        )

    def _create_wordmap_encoder(self):
        wordmap_enc = ValueEncoder(value_to_code=self.wordmap)
        wordmap_enc.set_default(len(self.wordmap))
        self._mappers.append(
            MapperSpec(field="tokens", target_field="tok_ids", encoder=wordmap_enc, preproc_fn=lambda x: x.text)
        )

    def _encode_for_batch(self, record):

        # if record.id in self._batch_cache:
        #     return self._batch_cache[record.id]

        def encode_seq(seq, encoder, preproc_fn=None):
            if preproc_fn is None:
                def preproc_fn(x):
                    return x
            # noinspection PyShadowingNames
            encoded = np.array([encoder[preproc_fn(w)] for w in seq], dtype=np.int32)
            return encoded

        def encode_item(item, encoder, preproc_fn=None):
            if preproc_fn is None:
                def preproc_fn(x):
                    return x
            # noinspection PyShadowingNames
            encoded = np.array(encoder[preproc_fn(item)], dtype=np.int32)
            return encoded

        output = {}

        for mapper in self._mappers:
            if mapper.field in record:
                if mapper.encoder_fn == "item":
                    enc_fn = encode_item
                elif mapper.encoder_fn == "seq":
                    enc_fn = encode_seq
                else:
                    enc_fn = mapper.encoder_fn
                assert isinstance(enc_fn, Callable), "encoder_fn should be either `item`/`seq` or a callable"

                encoded = enc_fn(
                    record[mapper.field], encoder=mapper.encoder,
                    preproc_fn=mapper.preproc_fn
                )

                if mapper.dtype is not None:
                    output[mapper.target_field] = encoded.astype(mapper.dtype)
                else:
                    output[mapper.target_field] = encoded

        num_tokens = len(record.tokens)

        output["lens"] = np.array(num_tokens, dtype=np.int32)
        output["id"] = record.id

        self._batch_cache[record.id] = output
        self._batch_cache.save()

        return output

    def collate(self, batch):
        if len(batch) == 0:
            return {}

        keys = batch[0].keys()

        def get_key(key):
            for sent in batch:
                yield sent[key]

        max_len = min(max(get_key("lens")), self._max_seq_len)

        def add_padding(encoded, pad):
            blank = np.ones((max_len,), dtype=np.int32) * pad
            blank[0:min(encoded.size, max_len)] = encoded[0:min(encoded.size, max_len)]
            return blank

        batch_o = {}

        for field in keys:
            if field == "lens" or field == "label":
                batch_o[field] = np.fromiter((min(i, max_len) for i in get_key(field)), dtype=np.int32)
            elif field == "id":
                batch_o[field] = np.fromiter(get_key(field), dtype=np.int64)
            elif field == "tokens" or field == "replacements":
                batch_o[field] = get_key(field)
            else:
                batch_o[field] = np.array(
                    [add_padding(item, self._default_padding[field]) for item in get_key(field)],
                    dtype=np.int64
                )

        # fbatch = defaultdict(list)

        # for sent in batch:
        #     for key, val in sent.items():
        #         fbatch[key].append(val)

        # max_len = max(fbatch["lens"])

        # batch_o = {}
        #
        # for field, items in fbatch.items():
        #     if field == "lens" or field == "label":
        #         batch_o[field] = np.array(items, dtype=np.int32)
        #     elif field == "id":
        #         batch_o[field] = np.array(items, dtype=np.int64)
        #     elif field == "tokens" or field == "replacements":
        #         batch_o[field] = items
        #     else:
        #         batch_o[field] = np.stack(items)[:, :max_len]

        return batch_o

        # return {
        #     key: np.stack(val)[:,:max_len] if key != "lens" and key != "replacements" and key != "tokens"
        #     else (np.array(val, dtype=np.int32) if key == "lens" else np.array(val)) for key, val in fbatch.items()}

    def generate_batches(self):
        batch = []
        if self._sort_by_length:
            records = self._iterate_sorted_by_length(limit_max_length=True)
        else:
            records = self._iterate_records(limit_max_length=True, shuffle=False)

        for id_ in records:
            encoded = self._batch_cache.get(id_, None)
            if encoded is None:
                encoded = self._encode_for_batch(self.get_record_with_id(id_))
            batch.append(encoded)
            if len(batch) >= self._batch_size:
                yield self.collate(batch)
                batch.clear()

        # for sent in records:
        #     batch.append(self._encode_for_batch(sent))
        #     if len(batch) >= self._batch_size:
        #         yield self.format_batch(batch)
        #         batch = []
        if len(batch) > 0:
            yield self.collate(batch)
        # yield self.format_batch(batch)

    def __iter__(self):
        self._batch_generator = self.generate_batches()
        return self

    def __next__(self):
        if self._batch_generator is None:
            raise StopIteration()
        return next(self._batch_generator)

    def __len__(self):
        total_valid = 0
        for id_ in self._iterate_record_ids():
            length = self._length_cache[id_]
            if length < self._max_seq_len:
                total_valid += 1
        return int(ceil(total_valid / self._batch_size))

    def tokenize_and_encode(self, text, annotations):
        record = self._prepare_record(0, text, annotations)
        encoded = self._encode_for_batch(record)
        return encoded


def test_batcher():
    from itertools import chain

    data = [
        "James B. Weaver (1833–1912) was a two-time candidate for US president and a congressman from Iowa.",
        "After serving in the Union Army in the Civil War, Weaver worked for the election of Republican candidates, but switched to the Greenback Party in 1877, and won election to the House in 1878.",
        "The Greenbackers nominated Weaver for president in 1880, but he received only 3.3 percent of the popular vote.",
        "He was again elected to the House in 1884 and 1886, where he worked for expansion of the money supply and for the opening of Indian Territory to white settlement.",
        "As the Greenback Party fell apart, he helped organize a new left-wing party, the Populists, and was their nominee for president in 1892.",
        "This time he gained 8.5 percent of the popular vote and won five states.",
        "Several of Weaver's political goals became law after his death, including the direct election of senators and a graduated income tax.",
    ]
    span_labels = [
        [[16, 69, "span1"]],
        [[39, 48, "span2"]],
        [[38, 47, "span3"]],
        [[61, 101, "span4"]],
        [[60, 69, "span5"]],
        [],
        [],
    ]
    sentence_labels = [
        "Good",
        "Bad",
        "Not so good",
        "Not so bad",
        "Moderate",
        "Moderate",
        "Moderate",
    ]

    nlp = Tokenizer()
    docs = [nlp(sent) for sent in data]
    words = set(t.text for t in chain(*docs))
    wordmap = dict(zip(words, range(len(words))))

    path = Path("temp_cache")
    path.mkdir(exist_ok=True, parents=True)

    # noinspection PyShadowingNames
    def iterate_data(data, span_labels, sentence_labels):
        for d, sl, l in zip(data, span_labels, sentence_labels):
            yield d, {"ents": sl, "cats": l}

    batcher = Batcher(
        list(iterate_data(data, span_labels, sentence_labels)),
        wordmap=wordmap,
        batch_size=2,
        max_seq_len=256,
        cache_dir=path
    )

    defaults = batcher.get_default_padding()

    num_batches = 0
    for ind, batch in enumerate(batcher):
        if ind == 0:
            # noinspection PyUnresolvedReferences
            assert batch["tok_ids"].shape[0] * batch["tok_ids"].shape[1] - (batch["tok_ids"] == defaults["tok_ids"]).sum() == 37
            # noinspection PyUnresolvedReferences
            assert batch["tags"].shape[0] * batch["tags"].shape[1] - (batch["tags"] == defaults["tags"]).sum() == 12
        num_batches += 1

    assert num_batches == 4


if __name__ == "__main__":
    test_batcher()
