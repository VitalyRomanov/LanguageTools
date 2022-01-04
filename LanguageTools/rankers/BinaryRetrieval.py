import json
import sys
from array import array
import mmap
import os
import pickle as p
from collections import Counter
from pathlib import Path

from more_itertools import windowed
from no_hassle_kv import CompactKeyValueStore
from psutil import virtual_memory
from tqdm import tqdm

from LanguageTools.Tokenizer import Doc
from LanguageTools.corpus.DocumentCorpus import DocumentCorpus
from LanguageTools.rankers import SimilarityEngine
from LanguageTools.rankers.utils import check_dir_exists


def dump_shard(path, id, postings_shard):
    post_name = os.path.join(path, "temp_shard_posting_{0:06d}".format(id))
    indx_name = os.path.join(path, "temp_shard_index_{0:06d}".format(id))

    index = dict()

    with open(post_name, "wb") as shard_sink:
        for key, postings in postings_shard.items():
            position = shard_sink.tell()
            written = shard_sink.write(p.dumps(postings, protocol=4))

            index[key] = (position, written)

    p.dump(index, open(indx_name, "wb"), protocol=4)

    return indx_name, post_name


def merge_shards(path, vocab, shards):

    shards_ = []
    terms = set()
    for index_path, shard_path in shards:
        shard_index = p.load(open(index_path, "rb"))
        f = open(shard_path, "r+b")
        mm = mmap.mmap(f.fileno(), 0)
        shards_.append((shard_index, mm))
        terms.update(shard_index.keys())

    index = PostingIndex(path)

    for t_id in tqdm(terms, desc="Postprocessing: ", leave=False):
        p_ = set()

        for s_ind, s_file in shards_:
            if t_id in s_ind:  # term never occurred in this shard
                pos_, len_ = s_ind[t_id]
                postings = p.loads(s_file[pos_: pos_+len_])  # shard store sets
                p_ |= postings

        doc_list = list(p_)
        doc_list.sort()
        index.add_posting(t_id, array("Q", doc_list))

    index.save()  # remove files

    for index_path, shard_path in shards:
        os.remove(index_path)
        os.remove(shard_path)

    return index


class BinaryRetrieval(SimilarityEngine):
    def __init__(self, corpus: DocumentCorpus, index_instantly=False, add_bigrams=False):
        self.path = corpus.path

        if not isinstance(corpus, DocumentCorpus):
            raise TypeError("corpus should be an instance of DocumentCorpus")

        super(BinaryRetrieval, self).__init__(corpus=corpus, add_bigrams=add_bigrams)

        self.inv_index = None
        if index_instantly:
            self.build_index()  # move this out of constructor

    @classmethod
    def posting_path(cls, corpus_path):
        return corpus_path.joinpath("postings")

    # def prepare_tokens_for_index(self, doc):
    #     return (tok.id for tok in doc)

    def prepare_tokens_for_index(self, doc):
        token_ids = []
        for tok in doc:
            token_id = tok.id
            assert token_id is not None, "token id is None, fatal error"
            token_ids.append(token_id)

        if self.add_bigrams:
            token_ids += self.get_bigrams(token_ids)

        return token_ids

    def build_index(self):
        # TODO
        # 1. include ngrams. alternative to ngrams is to use positional postings. however, ngram based implementation
        #   is simpler and allows to include misspell tolerant queries
        # 2. [v] (no need apparently) look into zarr
        # 3. [v] build disk-based index SPIMI see ref in 4.7
        # 4. compression (chapter 5)
        # 5. Correct orderings for ranked retrievals (chapter 6)
        # 6. Compare with SQLIte index

        shards = []
        shard_id = 0
        postings_shard = dict()

        for doc_ind, (id_, doc) in tqdm(
                enumerate(self.corpus.corpus),
                total=len(self.corpus.corpus), desc="Indexing: ", leave=False
        ):
            for token_id in self.prepare_tokens_for_index(doc):
                assert token_id is not None, "token id is None, fatal error"
                if token_id not in postings_shard:
                    postings_shard[token_id] = set()
                postings_shard[token_id].add(id_)

            # if self.add_bigrams:
            #     for bigram in self.into_bigrams(doc):
            #         if bigram not in postings_shard:
            #             postings_shard[bigram] = set()
            #         postings_shard[bigram].add(id_)

            if doc_ind % 1000 == 0:
                size = virtual_memory().available / 1024 / 1024  # total_size(postings_shard)
                # if size >= 1024*1024*1024: #1 GB
                if size <= 300:  # 100 MB
                    print(f"Only {size} mb of free RAM left")
                    shards.append(dump_shard(self.corpus.path, shard_id, postings_shard))

                    del postings_shard
                    postings_shard = dict()
                    shard_id += 1

        if len(postings_shard) > 0:
            shards.append(dump_shard(self.corpus.path, shard_id, postings_shard))

        self.merge_shards(shards)
        self.save()

    def merge_shards(self, shards):
        self.inv_index = merge_shards(self.posting_path(self.path), self.corpus.corpus.vocab, shards)

    def retrieve_sub_rank(self, tokens):

        doc_ranks = Counter()

        for t_id, rank in tokens:
            for doc_id in self.inv_index[t_id]:
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = 0.
                doc_ranks[doc_id] += rank

        return doc_ranks

    def retrieve_sub(self, tokens):
        # this does not work with query expansion
        empty = array("Q")

        doc_ranks = Counter()
        # for doc_id in set.intersection(*[self.inv_index.get(t_id, empty) for t_id, rank in self.filter_stopwords(tokens)]):
        for doc_id in self.doc_list_intersection(
                *[self.inv_index.get(t_id, empty) for t_id, rank in self.filter_stopwords(tokens)]
        ):
            # if [t[0] for t in tokens] in self.corpus.corpus[doc_id]:
            doc_ranks[doc_id] = 1.

        return doc_ranks

    @property
    def runtime_version(self):
        return f"python_{sys.version_info.major}.{sys.version_info.minor}"

    @property
    def class_name(self):
        return self.__class__.__name__

    def get_config_path(self):
        return self.posting_path(self.path).joinpath("config.json")

    def save_config(self):
        config = {
            "runtime_version": self.runtime_version,
            "class_name": self.class_name,
            "add_bigrams": self.add_bigrams,
        }

        with open(self.get_config_path(), "w") as config_sink:
            config_str = json.dumps(config, indent=4)
            config_sink.write(config_str)

    def read_config(self):
        with open(self.get_config_path(), "r") as config_sourse:
            config_str = config_sourse.read()
            config = json.loads(config_str)

            runtime_version = config.pop("runtime_version")
            class_name = config.pop("class_name")
            assert runtime_version == self.runtime_version
            assert class_name == self.class_name

            for name, val in config.items():
                setattr(self, name, val)

    def save(self):
        check_dir_exists(self.path)
        if self.inv_index is not None:
            self.inv_index.save()
        self.save_config()

    @classmethod
    def load(cls, path):
        path = Path(path)
        corpus = DocumentCorpus.load(path)
        inv_index = PostingIndex.load(cls.posting_path(path))

        retr = cls(corpus, index_instantly=False)
        retr.inv_index = inv_index
        retr.read_config()
        return retr


class BinaryRetrievalBiword(BinaryRetrieval):
    def __init__(self, *args, **kwargs):

        class BiwordVocabulary:
            def __init__(self):
                self.count = Counter()

            def add(self, biword_hash):
                self.count[biword_hash] += 1

            def most_common(self):
                return ((value, None, count) for value, count in self.count.most_common())

        self.biword_voc = BiwordVocabulary()

        super(BinaryRetrievalBiword, self).__init__(*args, **kwargs)

    def prepare_tokens_for_index(self, doc):
        token_ids = []
        for tok in doc:
            token_id = tok.id
            assert token_id is not None, "token id is None, fatal error"
            token_ids.append(token_id)

        bigram_ids = []
        for bigram in windowed(token_ids, 2):
            bigram_hash = self.get_hash(bigram)
            self.biword_voc.add(bigram_hash)
            bigram_ids.append(
                bigram_hash
            )
        return bigram_ids

    def merge_shards(self, shards):
        self.inv_index = merge_shards(self.posting_path(self.path), self.biword_voc, shards)

    def into_bigrams(self, doc):
        if isinstance(doc, Doc):
            token_ids = [tok.id for tok in doc]
        else:
            token_ids = [tok[0] for tok in doc]
        return self.get_bigrams(token_ids)

    def additional_processing(self, token_ids):
        return [(self.get_hash(bi), 1.) for bi in self.into_bigrams(token_ids)]


class PostingIndex(CompactKeyValueStore):

    def __init__(self, path, shard_size=2**30, **kwargs):
        super(PostingIndex, self).__init__(
            path=path, shard_size=shard_size,
            serializer=lambda x: x.tobytes(), deserializer=lambda x: array("Q", x)
        )

    def add_posting(self, term_id, postings):
        if self.index is None:
            raise Exception("Index is not initialized")

        self[term_id] = postings
        return term_id


def test_BinaryRetrieval():
    text_en = """<doc id="1300" url="https://en.wikipedia.org/wiki?curid=1300" title="Abalone">
    Abalone

    Abalone ( or ; via Spanish "", from the Rumsen language "aulón") is a common name for any of a group of small to very large sea snails, marine gastropod molluscs in the family Haliotidae.
    Other common names are ear shells, sea ears, and muttonfish or muttonshells in Australia, former in Great Britain, perlemoen in South Africa, and in New Zealand.
    S.W.A.T. M. D.
    """

    dcorpus = DocumentCorpus("dcorpus", "en")

    dcorpus.add_docs([text_en, "It is nice here. It's true."])

    ranker = BinaryRetrieval(dcorpus)

    assert list(ranker.query("Abalone").keys())[0] == 0
    assert list(ranker.query("It's true.").keys())[0] == 1


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path")
    args = parser.parse_args()

    dcorpus = DocumentCorpus.load(args.corpus_path)

    ranker = BinaryRetrieval(dcorpus, index_instantly=True)
    ranker.save()
