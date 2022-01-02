from array import array
from collections.abc import Iterable
from typing import Union

from more_itertools import windowed

from LanguageTools.corpus.DocumentCorpus import DocumentCorpus
from collections import Counter

from LanguageTools.utils.extra_utils import intersect_two_doc_lists


class SimilarityEngine:
    tok = None
    corpus = None
    _stopwords = None

    def __init__(self, corpus: DocumentCorpus, add_bigrams=False):
        self.corpus = corpus
        self.add_bigrams = add_bigrams
        self.prepare_stopwords()

    def prepare_stopwords(self):
        self._stopwords = set(self.corpus.corpus.vocab[word] for word in self.stopwords if word in self.corpus.corpus.vocab)

    def filter_stopwords(self, tokens):
        return [t for t in tokens if t not in self._stopwords]

    def get_bigrams(self, token_ids):
        bigram_ids = []
        for bigram in windowed(token_ids, 2):
            bigram_hash = self.get_hash(bigram)
            bigram_ids.append(
                bigram_hash
            )
        return bigram_ids

    def add_importance(self, token_ids):
        return [(tok, self.corpus.importance(tok)) for tok in token_ids]

    def get_hash(self, bigram):
        return hash(bigram) % 4294967296

    def query(self, q: Union[str, Iterable], exact=False, do_expansion=False, spellcheck=False, remove_stopwords=True):
        if isinstance(q, str):
            token_ids = []
            for t in self.corpus.corpus.tokenizer.token_gen(q):
                token_id = self.corpus.corpus.vocab.get(t.text, None)
                if token_id is not None:
                    token_ids.append(token_id)
            #     (
            #         self.corpus.corpus.vocab[t.text],  # token_id
            #         self.corpus.importance(t.text)  # token idf
            #     ) for t in self.corpus.corpus.tokenizer.token_gen(q)
            # ]
        elif isinstance(q, Iterable): # may be this is not what i think
            raise NotImplementedError()
            # token_ids = q
        else:
            raise TypeError("Type is not understood:", type(q))

        if spellcheck:
            pass  # generate several candidate queries

        original_query_tokens = token_ids
        if self.add_bigrams:
            token_ids = token_ids + self.get_bigrams(token_ids)

        token_ids = self.additional_processing(token_ids)

        if remove_stopwords:
            token_ids = self.filter_stopwords(token_ids)

        if not exact and do_expansion:
            token_ids = self.query_expansion(token_ids)

        token_ids = self.add_importance(token_ids)

        # spellchecker should produce several correction candidates (or maybe one)
        # here need to retrieve both queries and rank which query was more likely
        relevant_docs = self.retrieve(self.retrieve_sub(token_ids) if exact else self.retrieve_sub_rank(token_ids))

        if exact:
            relevant_docs = {
                key: val for key, val in relevant_docs.items() if original_query_tokens in self.corpus.get_as_token_ids(key)
            }
        return relevant_docs

    def additional_processing(self, token_ids):
        return token_ids

    def query_expansion(self, tokens):
        pass  # add extra words
        return tokens

    def retrieve(self, sub_doc_ranks):

        doc_ranks = Counter()

        for sub_doc_id, rank in sub_doc_ranks.items():
            doc_id = self.corpus.doc_sent.get_doc(sub_doc_id)
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = 0.
            doc_ranks[doc_id] += rank

        return doc_ranks

    @property
    def stopwords(self):
        return [
            "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out",
            "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into",
            "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the",
            "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
            "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to",
            "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have",
            "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can",
            "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself",
            "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by",
            "doing", "it", "how", "further", "was", "here", "than"
        ]

    def retrieve_sub(self, token_ids):
        return {}

    def retrieve_sub_rank(self, tokens):
        pass

    @staticmethod
    def doc_list_intersection(*arrays):
        num_arrays = len(arrays)
        if num_arrays == 0:
            current_target = array("Q")
        else:
            arrays = sorted(arrays, key=len)
            current_target = arrays[0]
            for arr in arrays[1:]:
                current_target = intersect_two_doc_lists(current_target, arr)

        return current_target

    def save(self):
        pass  # not implemented

    @classmethod
    def load(cls, path):
        pass  # not implemented
