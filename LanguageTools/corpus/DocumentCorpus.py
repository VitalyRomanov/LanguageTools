import os
import pickle as p
import sqlite3

from LanguageTools.Tokenizer import Sentencizer
from LanguageTools.corpus.TokenizedCorpus import TokenizedCorpus


class DocumentAssociationStore:
    def __init__(self, path):
        self.path = path

        self.db = sqlite3.connect(path)
        self.cur = self.db.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS association ("
                         "doc_id INTEGER NOT NULL, "
                         "sent_id INTEGER NOT NULL)")

        self.requires_commit = False

    def add(self, doc_id, sent_id):
        self.cur.execute(
            "INSERT INTO association (doc_id, sent_id) VALUES (?,?)",
            (doc_id, sent_id)
        )
        self.requires_commit = True

    def get_sents(self, doc_id):
        if self.requires_commit:
            self.commit()
        sents = self.cur.execute(f"SELECT sent_id FROM association WHERE doc_id = ?", (doc_id,)).fetchall()
        return [s[0] for s in sents]

    def get_doc(self, sent_id):
        if self.requires_commit:
            self.commit()
        return self.cur.execute(f"SELECT doc_id FROM association WHERE sent_id = ?", (sent_id,)).fetchone()[0]

    def commit(self):
        self.db.commit()
        self.requires_commit = False

    def __len__(self):
        if self.requires_commit:
            self.commit()
        return self.cur.execute("SELECT COUNT(DISTINCT doc_id) FROM association").fetchone()[0]

    def save(self):
        self.commit()


class DocumentCorpus:
    def __init__(self, path, lang, vocab=None, tokenizer=None, shard_size=2000000, freeze_vocab=False):
        self.path = path

        if not os.path.isdir(path):
            os.mkdir(path)

        self.corpus = TokenizedCorpus(
            path, vocab=vocab, tokenizer=tokenizer,
            shard_size=shard_size, freeze_vocab=freeze_vocab
        )

        self.sentencizer = None
        self.lang = lang

        self.doc_sent = DocumentAssociationStore(os.path.join(path, "doc_parts.db"))

        self.last_doc_id = len(self)

    def add_docs(self, docs, save_instantly=True):

        if self.sentencizer is None:
            self.sentencizer = Sentencizer(self.lang)

        for doc in docs:
            added = self.corpus.add_docs(self.sentencizer(doc), save_instantly=save_instantly)

            for a in added:
                self.doc_sent.add(self.last_doc_id, a)

            self.last_doc_id += 1

    def __iter__(self):
        self.iter_doc = 0
        return self

    def __next__(self):
        if self.iter_doc < len(self):
            parts = [self.corpus[id_] for id_ in self.doc_sent.get_sents(self.iter_doc)]
            self.iter_doc += 1
            return self.iter_doc - 1, parts
        else:
            raise StopIteration()

    def __len__(self):
        return len(self.doc_sent)

    def sent2doc(self, item):
        if isinstance(item, int):
            doc = self.doc_sent.get_doc(item)
            return doc
        else:
            docs = [self.doc_sent.get_doc(i) for i in item]
            return docs

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self.last_doc_id:
                raise IndexError("Document index out of range:", item)
            return [self.corpus[i] for i in self.doc_sent.get_sents(item)]

    def check_dir_exists(self):
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def importance(self, token):
        return 1.0

    def save(self):
        p.dump((
            self.path,
            self.lang,
            self.last_doc_id
        ), open(os.path.join(self.path, "doccorpus_params"), "wb"), protocol=4)
        self.doc_sent.commit()
        self.corpus.save()
        self.sentencizer = None

    @classmethod
    def load(cls, path):
        path, \
            lang, \
            last_doc_id = p.load(open(os.path.join(path, "doccorpus_params"), "rb"))

        doc_corpus = DocumentCorpus(path, lang)
        doc_corpus.last_doc_id = last_doc_id
        doc_corpus.corpus = TokenizedCorpus.load(path)

        return doc_corpus


def test_DocumentCospus():
    text_en = """<doc id="1300" url="https://en.wikipedia.org/wiki?curid=1300" title="Abalone">
Abalone

Abalone ( or ; via Spanish "", from the Rumsen language "aulón") is a common name for any of a group of small to very large sea snails, marine gastropod molluscs in the family Haliotidae.
Other common names are ear shells, sea ears, and muttonfish or muttonshells in Australia, former in Great Britain, perlemoen in South Africa, and in New Zealand.
S.W.A.T. M. D.
"""

    dcorpus = DocumentCorpus("dcorpus", "en")

    dcorpus.add_docs([text_en, "It is nice here. It's true."])

    # print(dcorpus[0])
    # print(dcorpus[1])

    dcorpus.save()

    dcorpus2 = DocumentCorpus.load("dcorpus")

    assert dcorpus[0] == dcorpus2[0]
    assert dcorpus[1] == dcorpus2[1]
    # print(dcorpus2[0])
    # print(dcorpus2[1])
