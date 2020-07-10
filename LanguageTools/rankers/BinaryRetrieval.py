import os
import pickle
from collections import Counter

from LanguageTools.DocumentCorpus import DocumentCorpus
from LanguageTools.rankers import SimilarityEngine
from LanguageTools.file_utils import check_dir_exists


class BinaryRetrieval(SimilarityEngine):
    def __init__(self, corpus: DocumentCorpus):
        if not isinstance(corpus, DocumentCorpus):
            raise TypeError("courpus should be an instance of DocumentCorpus")

        super(BinaryRetrieval, self).__init__(corpus=corpus)

        self.inv_index = None
        self.build_index()

    def build_index(self):
        # TODO
        # 1. include ngrams. alternative to ngrams is to use positional postings. hoewver, ngram based implementation
        #   is simpler and allows to include misspell tolerant queries
        # 2. look into zarr
        # 3. build disk-based index SPIMI see ref in 4.7
        # 4. compression (chapter 5
        # 5. Correct orderings for ranked retrievals (chapter 6)
        self.inv_index = dict()

        for id_, doc in self.corpus.corpus:
            for token in doc:
                assert token.id is not None, "token id is None, fatal error"
                if token.id not in self.inv_index:
                    self.inv_index[token.id] = set()
                self.inv_index[token.id].add(id_)

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

    def save(self, path):
        check_dir_exists(path)

        pickle.dump(self.inv_index, open(os.path.join(path, "inv_index"), "wb"))
        pickle.dump(self.corpus.path, open(os.path.join(path, "corpus_ref"), "wb"))

    @classmethod
    def load(cls, path):
        corpus = DocumentCorpus.load(pickle.load(open(os.path.join(path, "corpus_ref"), "rb")))
        retr = BinaryRetrieval(corpus)
        retr.inv_index = pickle.load(open(os.path.join(path, "inv_index"), "rb"))
        return retr


