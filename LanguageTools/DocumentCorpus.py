from LanguageTools.TokenizedCorpus import TokenizedCorpus
from LanguageTools.Tokenizer import Sentencizer
import os
import pickle as p

class DocumentCorpus:
    def __init__(self, path, lang):
        self.path = path
        self.corpus = TokenizedCorpus(path)
        self.sentencizer = Sentencizer(lang)
        self.lang = lang

        self.sent_to_doc = dict()
        self.doc_to_sent = dict()
        self.last_doc_id = 0

    def add_docs(self, docs, save_instantly=True):

        for doc in docs:
            added = self.corpus.add_docs(self.sentencizer(doc), save_instantly=save_instantly)

            for a in added:
                self.sent_to_doc[a] = self.last_doc_id

            self.doc_to_sent[self.last_doc_id] = added

            self.last_doc_id += 1

    def sent2doc(self, item):
        if isinstance(item, int):
            return self.sent_to_doc[item]
        else:
            return [self.sent_to_doc[i] for i in item]

    def __getitem__(self, item):
        if isinstance(item, int):
            return [self.corpus[i] for i in self.doc_to_sent[item]]

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
            self.last_doc_id
        ), open(os.path.join(self.path, "docindex"), "wb"), protocol=4)
        self.corpus.save()

    @classmethod
    def load(cls, path):
        path, \
            lang, \
            sent_to_doc, \
            last_doc_id = p.load(open(os.path.join(path, "docindex"), "rb"))

        doc_corpus = DocumentCorpus(path, lang)
        doc_corpus.sent_to_doc = sent_to_doc
        doc_corpus.last_doc_id = last_doc_id

        return doc_corpus





