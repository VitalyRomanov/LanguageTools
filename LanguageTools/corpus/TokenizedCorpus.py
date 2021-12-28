# from gensim.corpora import MmCorpus # does not preserve the order
import logging
import os
import pickle as p
import types
from typing import Iterable

from no_hassle_kv import KVStore

from LanguageTools.Tokenizer import Token, Doc
from LanguageTools.Tokenizer import Tokenizer, Sentencizer
from LanguageTools.Vocabulary import SqliteVocabulary


# def isiterable(obj):
#     try:
#         iter(obj)
#     except Exception:
#         return False
#     else:
#         return True

# def into_string(tokens):
#     out = ""
#     for t in tokens:
#         out += t.text
#         if t.tailspace:
#             out += " "
#     return out


class TokenizedCorpus(KVStore):
    """
    Creates an on-disk storage for text corpus. Each text unit can be retrieved using an iterator or with index slices.
    Corpus is stored in shards.
    """
    vocab = None
    tok = None
    persist_tokenizer = False
    file_index = None
    index = None

    def __init__(self, path, vocab=None, tokenizer=None, shard_size=2000000, freeze_vocab=False):
        """
        Initialize corpus storage
        :param path: path where data will be stored
        :param vocab: Vocabulary should be an instance of LanguageTools.Vocabulary
        :param tokenizer: provide an instance of LanguageTools.Tokenizer with custom tokenization rules
        :param shard_size: shard size in bytes
        :param freeze_vocab: whether to add new words to vocabulary
        """
        super(TokenizedCorpus, self).__init__(path=path, shard_size=shard_size)

        self._freeze_vocab = freeze_vocab

        if not vocab:
            self.vocab = SqliteVocabulary(os.path.join(path, "voc.db"))
            self._freeze_vocab = False
        else:
            self.vocab = vocab

        if tokenizer:
            self.tok = tokenizer
            self.persist_tokenizer = True

        self.new_doc_id = len(self.index)

        self.save_instantly_warn = True

    @property
    def tokenizer(self):
        if self.tok is None:
            self.tok = Tokenizer()
        return self.tok

    def freeze_vocab(self):
        self._freeze_vocab = True

    def add_docs(self, docs, save_instantly=True):

        if self.save_instantly_warn and save_instantly:
            logging.warning("Set `save_instantly` to False for performance")
            self.save_instantly_warn = False

        if self.tok is None:
            self.tok = Tokenizer()

        added = [] # bad, high mem allocation

        for doc in docs:
            if isinstance(doc, str):
                tokenized_doc = list(self.tok.token_gen(doc))
            elif isinstance(doc, list):
                tokenized_doc = doc
            elif isinstance(doc, types.GeneratorType):
                tokenized_doc = list(doc)
            else:
                raise TypeError("Following types are accepted: str | List[Token] | GeneratorType[Token]. Instead received:", type(doc))
            added.append(self.add_doc(tokenized_doc))

        if save_instantly:
            if len(added):
                self.save_index()
                self.save_vocab()
                self.save_param()

        return added

    def add_doc(self, doc):
        if not self._freeze_vocab:
            for token in doc:
                self.vocab.add_token(token.text)

        transformed_doc = [(self.vocab[t.text], t.tailspace) for t in doc]

        doc_id = self.new_doc_id #len(self.index)
        self.new_doc_id += 1

        self[doc_id] = transformed_doc
        return doc_id

    def __len__(self):
        # return len(self.index)
        return self.new_doc_id

    def __getitem__(self, doc_id):
        if isinstance(doc_id, int):
            return Doc(self.wrap_into_token(self.get_with_id(doc_id)))
        elif isinstance(doc_id, slice):
            return (Doc(self.wrap_into_token(self.get_with_id(i))) for i in range(doc_id.start, doc_id.stop, doc_id.step))
        elif isinstance(doc_id, Iterable):
            return (Doc(self.wrap_into_token(self.get_with_id(i))) for i in doc_id)
        else:
            ValueError("Format not understood: doc_id can be int, slice, or iterable but found ", type(doc_id))

    def __iter__(self):
        self.iter_doc = 0
        return self

    def __next__(self):
        if self.iter_doc < len(self):
            c_doc_id = self.iter_doc
            self.iter_doc += 1
            return c_doc_id, self[c_doc_id]
        else:
            raise StopIteration()

    def wrap_into_token(self, tokens):
        return (Token(tailspace=t[1], text=self.vocab[t[0]], id=t[0]) for t in tokens)

    def save_corpus_param(self):
        if not self.persist_tokenizer:
            self.tok = None

        p.dump((
            self.tok,
            self.persist_tokenizer,
            self.new_doc_id,
            self._freeze_vocab,
        ), open(os.path.join(self.path, "corpus_params"), "wb"), protocol=4)

    def load_param(self):
        super(TokenizedCorpus, self).load_param()

        self.tok, \
            self.persist_tokenizer, \
            self.new_doc_id, \
            self._freeze_vocab = p.load(open(os.path.join(self.path, "corpus_params"), "rb"))

    def save_vocab(self):
        # self.vocab.save(os.path.join(self.path, "vocab"))
        self.vocab.save()

    def load_vocab(self):
        # self.vocab = Vocabulary.load(os.path.join(self.path, "vocab"))
        pass

    def save_index(self):
        super(TokenizedCorpus, self).save_index()
        # self.index.commit()
        # p.dump(self.index, open(os.path.join(self.path, "index"), "wb"), protocol=4)

    def load_index(self):
        super(TokenizedCorpus, self).load_index()
        pass
        # self.index = p.load(open(os.path.join(self.path, "index"), "rb"))

    def save(self):
        super(TokenizedCorpus, self).save()
        self.save_vocab()
        self.save_corpus_param()

    @classmethod
    def load(cls, path):
        tcorpus = cls(path)
        tcorpus.load_param()
        tcorpus.load_vocab()
        tcorpus.load_index()
        return tcorpus

    # def __del__(self):
    #     self.commit()
    #     self.close()

def test_TokenizedCorpus():
    text_en = """<doc id="1300" url="https://en.wikipedia.org/wiki?curid=1300" title="Abalone">
Abalone

Abalone ( or ; via Spanish "", from the Rumsen language "aulón") is a common name for any of a group of small to very large sea snails, marine gastropod molluscs in the family Haliotidae.
Other common names are ear shells, sea ears, and muttonfish or muttonshells in Australia, former in Great Britain, perlemoen in South Africa, and in New Zealand.
S.W.A.T. M. D.
"""
    sentencizer_en = Sentencizer("en")

    sents = sentencizer_en(text_en)

    tcorpus = TokenizedCorpus("tcorpus")

    tcorpus.add_docs(s for s in sents)

    # print(into_string(tcorpus[0]))
    # print(into_string(tcorpus[1]))

    tcorpus.save()

    tcorpus2 = TokenizedCorpus.load("tcorpus")

    # print(tcorpus2[0].tokens)
    # print(tcorpus2[1])

    assert tcorpus[0] == tcorpus2[0]
    assert tcorpus[1] == tcorpus2[1]
