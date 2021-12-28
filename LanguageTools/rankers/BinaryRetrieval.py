import mmap
import os
import pickle as p
from collections import Counter

from no_hassle_kv import KVStore
from psutil import virtual_memory
from tqdm import tqdm

from LanguageTools.corpus.DocumentCorpus import DocumentCorpus
from LanguageTools.rankers import SimilarityEngine
from LanguageTools.rankers.utils import check_dir_exists


def dump_shard(path, id, shard_postings):
    post_name = os.path.join(path, "temp_shard_posting_{0:06d}".format(id))
    indx_name = os.path.join(path, "temp_shard_index_{0:06d}".format(id))

    index = dict()

    with open(post_name, "wb") as shard_sink:
        for key, postings in shard_postings.items():
            position = shard_sink.tell()
            written = shard_sink.write(p.dumps(postings, protocol=4))

            index[key] = (position, written)

    p.dump(index, open(indx_name, "wb"), protocol=4)

    return indx_name, post_name


def merge_shards(path, vocab, shards):

    shards_ = []
    for index_path, shard_path in shards:
        shard_index = p.load(open(index_path, "rb"))
        f = open(shard_path, "r+b")
        mm = mmap.mmap(f.fileno(), 0)
        shards_.append((shard_index, mm))

    index = PostingIndex(path)

    for t_id, _, _ in tqdm(vocab.most_common(), desc="Postprocessing: ", leave=False):
        p_ = set()

        for s_ind, s_file in shards_:
            if t_id in s_ind:  # term never occurred in this shard
                pos_, len_ = s_ind[t_id]
                postings = p.loads(s_file[pos_: pos_+len_])  # shard store sets
                p_ |= postings

        index.add_posting(t_id, p_)

    index.save()  # remove files

    for index_path, shard_path in shards:
        os.remove(index_path)
        os.remove(shard_path)

    return index


class BinaryRetrieval(SimilarityEngine):
    def __init__(self, corpus: DocumentCorpus, index_instantly=False):
        self.path = corpus.path

        if not isinstance(corpus, DocumentCorpus):
            raise TypeError("corpus should be an instance of DocumentCorpus")

        super(BinaryRetrieval, self).__init__(corpus=corpus)

        self.inv_index = None
        if index_instantly:
            self.build_index()  # move this out of constructor

    @classmethod
    def posting_path(cls, corpus_path):
        return os.path.join(corpus_path, "postings")

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
        shard_postings = dict()

        for doc_ind, (id_, doc) in tqdm(
                enumerate(self.corpus.corpus),
                total=len(self.corpus.corpus), desc="Indexing: ", leave=False
        ):
            for token in doc:
                assert token.id is not None, "token id is None, fatal error"
                if token.id not in shard_postings:
                    shard_postings[token.id] = set()
                shard_postings[token.id].add(id_)

            if doc_ind % 1000 == 0:
                size = virtual_memory().available / 1024 / 1024  # total_size(shard_postings)
                # if size >= 1024*1024*1024: #1 GB
                if size <= 300:  # 100 MB
                    print(f"Only {size} mb of free RAM left")
                    shards.append(dump_shard(self.corpus.path, shard_id, shard_postings))

                    del shard_postings
                    shard_postings = dict()
                    shard_id += 1

        if len(shard_postings) > 0:
            shards.append(dump_shard(self.corpus.path, shard_id, shard_postings))

        self.inv_index = merge_shards(self.posting_path(self.path), self.corpus.corpus.vocab, shards)
        self.save()

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

        doc_ranks = Counter()
        for doc_id in set.intersection(*[self.inv_index[t_id] for t_id, rank in tokens]):
            doc_ranks[doc_id] = 1.

        return doc_ranks

    def save(self):
        check_dir_exists(self.path)
        if self.inv_index is not None:
            self.inv_index.save()

    @classmethod
    def load(cls, path):
        corpus = DocumentCorpus.load(p.load(open(path, "rb")))
        inv_index = PostingIndex.load(p.load(open(BinaryRetrieval.posting_path(path), "rb")))

        retr = BinaryRetrieval(corpus, index_instantly=False)
        retr.inv_index = inv_index
        return retr


class PostingIndex(KVStore):

    def __init__(self, path, shard_size=2**30):
        super(PostingIndex, self).__init__(path=path, shard_size=shard_size)

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
