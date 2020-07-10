# from gensim.corpora import MmCorpus # does not preserve the order
from LanguageTools.Vocabulary import Vocabulary
from LanguageTools.Tokenizer import Tokenizer, Sentencizer
from LanguageTools.Tokenizer import Token, Doc
import pickle as p
import mmap
import os, sys
import json
import types

def isiterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def into_string(tokens):
    out = ""
    for t in tokens:
        out += t.text
        if t.tailspace:
            out += " "
    return out

class TokenizedCorpus:
    vocab = None
    file_index = None
    index = None

    def __init__(self, path, vocab=None, tokenizer=None, shard_size=2000000, freeze_vocab=False):

        self.freeze_vocab = freeze_vocab

        if not vocab:
            self.vocab = Vocabulary()
            self.freeze_vocab = False
        else:
            self.vocab = vocab

        self.tok = tokenizer if tokenizer else Tokenizer()

        self.file_index = dict() # (shard, filename)
        self.index = dict() # (shard, position, length)

        self.opened_shards = dict() # (shard, file, mmap object) if mmap is none -> opened for write

        self.shard_for_write = 0
        self.written_in_current_shard = 0
        self.shard_size=shard_size

        self.path = path


    def add_docs(self, docs, save_instantly=True):

        len_index = len(self.index)

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
            if len(self.index) != len_index:
                self.save_index()
                self.save_vocab()
                self.save_param()

        return added

    def add_doc(self, doc):
        if not self.freeze_vocab:
            for token in doc:
                self.vocab.add_token(token.text)

        transformed_doc = [(self.vocab[t.text], t.tailspace) for t in doc]
        # switch to numpy's structured array?
        # it appears that pickled structured array occupies about three times as much space as list of tuples
        serialized_doc = p.dumps(transformed_doc, protocol=4)

        doc_id = len(self.index)

        f, _ = self.writing_mode(self.shard_for_write)

        position = f.tell()
        written = f.write(serialized_doc)
        self.index[doc_id] = (self.shard_for_write, position, written)
        self.increment_doc_count()
        return doc_id

    def increment_doc_count(self):
        self.written_in_current_shard += 1
        if self.written_in_current_shard >= self.shard_size:
            self.shard_for_write += 1
            self.written_in_current_shard = 0

    def __getitem__(self, doc_id):
        if isinstance(doc_id, int):
            return Doc(self.wrap_into_token(self.get_with_id(doc_id)))
        elif isinstance(doc_id, slice):
            return (Doc(self.wrap_into_token(self.get_with_id(i))) for i in range(doc_id.start, doc_id.stop, doc_id.step))
        elif isiterable(doc_id):
            return (Doc(self.wrap_into_token(self.get_with_id(i))) for i in doc_id)
        else:
            ValueError("Format not understood: doc_id can be int, slice, or iterable but found ", type(doc_id))

    def __iter__(self):
        self.iter_doc = 0
        return self

    def __next__(self):
        if self.iter_doc >= len(self.index):
            raise StopIteration()
        c_doc_id = self.iter_doc
        self.iter_doc += 1
        return c_doc_id, self[c_doc_id]

    def wrap_into_token(self, tokens):
        return (Token(tailspace=t[1], text=self.vocab[t[0]], id=t[0]) for t in tokens)

    def get_with_id(self, doc_id):
        shard, pos, len_ = self.index[doc_id]
        _, mm = self.reading_mode(shard)
        return p.loads(mm[pos: pos+len_])

    def get_name_format(self, id_):
        return 'shard_{0:04d}'.format(id_)

    def open_for_read(self, name):
        # raise filenotexists
        f = open(os.path.join(self.path, name), "r+b")
        mm = mmap.mmap(f.fileno(), 0)
        return f, mm

    def open_for_write(self, name):
        # raise filenotexists
        self.check_dir_exists()
        f = open(os.path.join(self.path, name), "ab")
        return f, None

    def check_dir_exists(self):
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def writing_mode(self, id_):
        if id_ not in self.opened_shards:
            if id_ not in self.file_index:
                self.file_index[id_] = self.get_name_format(id_)
            self.opened_shards[id_] = self.open_for_write(self.file_index[id_])
        elif self.opened_shards[id_][1] is not None: # mmap is None
            self.opened_shards[id_][1].close()
            self.opened_shards[id_][0].close()
            self.opened_shards[id_] = self.open_for_write(self.file_index[id_])
        return self.opened_shards[id_]

    def reading_mode(self, id_):
        if id_ not in self.opened_shards:
            if id_ not in self.file_index:
                self.file_index[id_] = self.get_name_format(id_)
            self.opened_shards[id_] = self.open_for_read(self.file_index[id_])
        elif self.opened_shards[id_][1] is None:
            self.opened_shards[id_][0].close()
            self.opened_shards[id_] = self.open_for_read(self.file_index[id_])
        return self.opened_shards[id_]

    def save_param(self):
        p.dump((
            self.freeze_vocab,
            self.file_index,
            self.shard_for_write,
            self.written_in_current_shard,
            self.shard_size,
            self.path
        ), open(os.path.join(self.path, "params"), "wb"), protocol=4)

    def load_param(self):
        self.freeze_vocab,\
        self.file_index,\
        self.shard_for_write,\
        self.written_in_current_shard,\
        self.shard_size,\
        self.path = p.load(open(os.path.join(self.path, "params"), "rb"))

    def save_vocab(self):
        self.vocab.save(os.path.join(self.path, "vocab"))

    def load_vocab(self):
        self.vocab = Vocabulary.load(os.path.join(self.path, "vocab"))

    def save_index(self):
        p.dump(self.index, open(os.path.join(self.path, "index"), "wb"), protocol=4)

    def load_index(self):
        self.index = p.load(open(os.path.join(self.path, "index"), "rb"))

    def save(self):
        self.save_index()
        self.save_vocab()
        self.save_param()

    @classmethod
    def load(cls, path):
        tcorpus = TokenizedCorpus(path)
        tcorpus.load_param()
        tcorpus.load_vocab()
        tcorpus.load_index()
        return tcorpus


    def close_all_shards(self):
        for shard in self.opened_shards.values():
            for s in shard:
                if s:
                    s.close()




if __name__=="__main__":
    text_en = """<doc id="1300" url="https://en.wikipedia.org/wiki?curid=1300" title="Abalone">
Abalone

Abalone ( or ; via Spanish "", from the Rumsen language "aulón") is a common name for any of a group of small to very large sea snails, marine gastropod molluscs in the family Haliotidae.
Other common names are ear shells, sea ears, and muttonfish or muttonshells in Australia, former in Great Britain, perlemoen in South Africa, and in New Zealand.
S.W.A.T. M. D.
"""
    sentencizer_en = Sentencizer("en")
    tokenizer = Tokenizer()

    sents = sentencizer_en(text_en)

    tcorpus = TokenizedCorpus("tcorpus")

    tcorpus.add_docs(s for s in sents)

    print(into_string(tcorpus[0]))
    print(into_string(tcorpus[1]))

    # for t in tcorpus[0]:
    #     print(t, end=" ")
    # print()
    #
    # for t in tcorpus[1]:
    #     print(t, end=" ")
    # print()

    tcorpus.close_all_shards()

    tcorpus2 = TokenizedCorpus.load("tcorpus")

    print(tcorpus2[0].tokens)
    print(tcorpus2[1])

    # for t in tcorpus2[0]:
    #     print(t, end=" ")
    # print()
    #
    # for t in tcorpus2[1]:
    #     print(t, end=" ")
    # print()