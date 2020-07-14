from LanguageTools.TokenizedCorpus import TokenizedCorpus
from LanguageTools.Tokenizer import Sentencizer
from LanguageTools.utils import CompactStorage
import os
import pickle as p
import numpy as np

class DocumentCorpus:
    def __init__(self, path, lang):
        self.path = path
        self.corpus = TokenizedCorpus(path)
        self.sentencizer = Sentencizer(lang)
        self.lang = lang

        # self.sent_to_doc = dict()
        # self.doc_to_sent = dict()
        self.sent_to_doc = CompactStorage(1, 100000)
        self.doc_to_sent = CompactStorage(2, 1000)
        # self.init_sent_to_doc(100000)
        # self.init_doc_to_sent(1000)

        self.last_doc_id = 0

    # def init_sent_to_doc(self, size):
    #     self.sent_to_doc = np.zeros(shape=(size,), dtype=np.uint32)
    #
    # def init_doc_to_sent(self, size):
    #     self.doc_to_sent = np.zeros(shape=(size,2), dtype=np.uint32)
    #
    # def resize_sent_to_doc(self, new_size):
    #     self.sent_to_doc.resize((new_size,))
    #
    # def resize_doc_to_sent(self, new_size):
    #     self.doc_to_sent.resize((new_size, 2))

    def add_docs(self, docs, save_instantly=True):

        for doc in docs:
            added = self.corpus.add_docs(self.sentencizer(doc), save_instantly=save_instantly)

            # if len(self.corpus) >= self.sent_to_doc.shape[0]:
            #     self.resize_sent_to_doc(int(self.sent_to_doc.shape[0] * 1.2)) # conservative growth

            for a in added:
                assert a == len(self.sent_to_doc)
                self.sent_to_doc.append(self.last_doc_id)
                # self.sent_to_doc[a] = self.last_doc_id

            # if self.last_doc_id >= self.doc_to_sent.shape[0]:
            #     self.resize_doc_to_sent(int(self.doc_to_sent.shape[0] * 1.2)) # conservative growth

            assert self.last_doc_id == len(self.doc_to_sent)
            self.doc_to_sent.append((added[0], added[-1]+1))
            # self.doc_to_sent[self.last_doc_id, 0] = added[0]
            # self.doc_to_sent[self.last_doc_id, 1] = added[-1] + 1 # need to add one to use as argument for range()

            self.last_doc_id += 1

    def sent2doc(self, item):
        if isinstance(item, int):
            return self.sent_to_doc[item]
        else:
            return [self.sent_to_doc[i] for i in item]

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self.last_doc_id:
                raise IndexError("Document index out of range:", item)
            return [self.corpus[i] for i in range(*self.doc_to_sent[item])]

    def check_dir_exists(self):
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def importance(self, token):
        return 1.0

    def save(self):
        p.dump((
            self.path,
            self.lang,
            self.sent_to_doc,
            self.doc_to_sent,
            self.last_doc_id
        ), open(os.path.join(self.path, "docindex"), "wb"), protocol=4)
        self.corpus.save()

    @classmethod
    def load(cls, path):
        path, \
            lang, \
            sent_to_doc, \
            doc_to_sent, \
            last_doc_id = p.load(open(os.path.join(path, "docindex"), "rb"))

        doc_corpus = DocumentCorpus(path, lang)
        doc_corpus.sent_to_doc = sent_to_doc
        doc_corpus.doc_to_sent = doc_to_sent
        doc_corpus.last_doc_id = last_doc_id
        doc_corpus.corpus = TokenizedCorpus.load(path)

        return doc_corpus





