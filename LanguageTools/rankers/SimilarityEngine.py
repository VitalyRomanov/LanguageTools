from collections.abc import Iterable
from LanguageTools.corpus.DocumentCorpus import DocumentCorpus
from LanguageTools.Tokenizer import Tokenizer
from collections import Counter


class SimilarityEngine:
    tok = None
    corpus = None

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus

    def query(self, q, exact=False, do_expansion=False, spellcheck=False):
        if isinstance(q, str):
            token_ids = [(self.corpus.corpus.vocab[t.text], self.corpus.importance(t.text)) for t in self.corpus.corpus.tokenizer.token_gen(q)]
        if isinstance(q, Iterable): # may be this is not what i think
            pass
        else:
            raise TypeError("Type is not understood:", type(q))

        if spellcheck:
            pass # generate several candidate queries

        if not exact and do_expansion :
            token_ids = self.query_expansion(token_ids)

        # spellchecker should produce several correction candidates (or maybe one)
        # here need to retrieve both queries and rank which query was more likely
        relevant_docs = self.retrieve(self.retrieve_sub(token_ids) if exact else self.retrieve_sub_rank(token_ids))
        return relevant_docs

    def query_expansion(self, tokens):
        pass # add extra words
        return tokens

    def retrieve(self, sub_doc_ranks):

        doc_ranks = Counter()

        for sub_doc_id, rank in sub_doc_ranks.items():
            doc_id = self.corpus.doc_sent.get_doc(sub_doc_id)
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = 0.
            doc_ranks[doc_id] += rank

        return doc_ranks

    def retrieve_sub(self, token_ids):
        return {}

    def retrieve_sub_rank(self, tokens):
        pass

    def save(self):
        pass # not implemented

    @classmethod
    def load(cls):
        pass # not implemented