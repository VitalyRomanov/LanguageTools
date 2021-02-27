import pickle as p
import sqlite3
from collections import Counter

import numpy as np
from numpy import array
from numpy.random import choice


class Vocabulary:

    def __init__(self):
        self.count = Counter()
        self.ids = {}
        self.inv_ids = {}
        self.prob_valid = False

        self.add(['UNK'])


    def add(self, tokens):
        for token in tokens:
            self.add_token(token)

        self.prob_valid = False

    def add_token(self, token):
        if token in self.ids:
            self.count[self.ids[token]] += 1
        else:
            new_id = len(self.ids)
            self.ids[token] = new_id
            self.inv_ids[new_id] = token
            self.count[new_id] = 1

    def drop_oov(self, tokens):
        return (self.is_oov(t) for t in tokens)

    def is_oov(self, token):
        return token in self.ids

    @property
    def total_words(self):
        if not self.prob_valid:
            self.calculate_prob()
        return self.word_count_sum

    def tokens2ids(self, tokens):
        return [self.ids.get(token, 0) for token in tokens]

    def ids2tokens(self, ids):
        return [self.inv_ids.get(token_id, 'UNK') for token_id in ids]

    def __len__(self):
        return len(self.count)

    def __getitem__(self, item):
        if type(item) == str:
            return self.ids[item]
        elif type(item) == int:
            return self.inv_ids[item]
        else:
            raise KeyError("")

    def most_common(self, length=None):
        if length is None:
            length = len(self)
        return [(token_id, self.inv_ids[token_id], freq) for token_id, freq in self.count.most_common(length)]

    def values(self):
        return self.count.values()

    def calculate_prob(self):
        p = array(list(self.count.values()))
        self.word_count_sum = np.sum(p)
        self.p = p / self.word_count_sum
        self.prob_valid = True

    def sample(self,n_samples, limit_top=-1):
        # make sure that counts follow the order of ids

        if limit_top == -1:
            limit_top = len(self.count)

        if not self.prob_valid:
            self.calculate_prob()

        if limit_top!=-1:
            p = self.p[:limit_top] / np.sum(self.p[:limit_top])
        else:
            p = self.p

        return list(choice(limit_top, size=n_samples, p=p))

    def save(self, destination):
        p.dump(self, open(destination, "wb"))
        
        # with open(destination, "w", encoding="utf8") as voc_sink:
        #     voc_sink.write("Version: %d\n" % time())
        #     for token, token_id in self.ids.items():
        #         voc_sink.write("%d\t%s\t%d\n" % (token_id, token, self.count[token_id]))



    @classmethod
    def load(self, location):

        # voc = Vocabulary()
        # with open(location, "r", encoding="utf8") as voc_source:
        #     for ind, line in enumerate(voc_source):
        #         if ind == 0 or ind == 1: continue
        #         if line.strip():
        #             token_id, token, freq = line.strip().split()
        #             voc.ids[token] = token_id
        #             voc.inv_ids[token_id] = token
        #             voc.count[token_id] = int(freq)
        #
        # voc.calculate_prob()
        # return voc
        return p.load(open(location, "rb"))


# class PersistentVocabulary(Vocabulary):
#     def __init__(self, path, writeback=False):
#         self.path = path
#
#         self.count = Counter()
#         self.ids = StringTrie()# shelve.open(os.path.join(self.path, "vocab_ids"), protocol=4, writeback=writeback)
#         self.inv_ids = dict() #shelve.open(os.path.join(self.path, "vocab_inv_ids"), protocol=4, writeback=writeback)
#
#         self.prob_valid = False
#
#         self.add(['UNK'])
#
#     # def add_token(self, token):
#     #     if token in self.ids:
#     #         self.count[self.ids[token]] += 1
#     #     else:
#     #         new_id = len(self.ids)
#     #         self.ids[token] = new_id
#     #         self.inv_ids[str(new_id)] = token
#     #         self.count[new_id] = 1
#     #
#     # def __getitem__(self, item):
#     #     if type(item) == str:
#     #         return self.ids[item]
#     #     elif type(item) == int:
#     #         return self.inv_ids[str(item)]
#     #     else:
#     #         raise KeyError("")
#
#     # def save(self, *args):
#     #     self.ids.sync()
#     #     self.inv_ids.sync()
#     #
#     #     p.dump((
#     #         self.count, self.prob_valid
#     #     ), open(os.path.join(self.path, "vocab_params"), "wb"))
#     #
#     # def load(self, path):
#     #     voc = PersistentVocabulary(path)
#     #     self.count, self.prob_valid = p.load(open(os.path.join(path, "vocab_params"), "rb"))
#
#     def __del__(self):
#         self.ids.close()
#         self.inv_ids.close()

class CacheFull(Exception):
    def __init__(self, *args, **kwargs):
        super(CacheFull, self).__init__(*args, **kwargs)

class SimpleCache:
    def __init__(self, size):
        self.size = size
        self.cache = dict()
        self.count = Counter()

    def __getitem__(self, item):
        value = self.cache[item]
        self.count[item] += 1
        return value

    def __setitem__(self, key, value):
        self.cache[key] = value

        if len(self.cache) >= self.size:
            raise CacheFull

        self.count[key] += 1

    def __contains__(self, item):
        return item in self.cache

    def free(self, fraction=0.8):
        # self.cache = dict()
        # self.count = Counter()
        total = sum(self.count.values())
        accum = 0

        for key, count in reversed(self.count.most_common()):
            accum += count
            if accum <= fraction * total:
                del self.cache[key]
                del self.count[key]
            else:
                self.count[key] = 0

    def __iter__(self):
        return iter(self.cache.items())


class SqliteVocabulary(Vocabulary):
    def __init__(self, path, cache_size=2000000):
        # super(SqliteVocabulary, self).__init__()

        self.path = path

        self.db = sqlite3.connect(path)
        self.cur = self.db.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS [vocabulary] ("
                        "[word] TEXT PRIMARY KEY NOT NULL UNIQUE, "
                        "[id] INTEGER NOT NULL UNIQUE, "
                        "[count] INTEGER NOT NULL)")
        self.build_inv_index()

        self.requires_commit = False
        self.need_inv_index = False
        self.newid = next(iter(self.cur.execute("select count(distinct word) from vocabulary")))[0]

        self.cache = SimpleCache(cache_size)

    def __contains__(self, item):
        if item in self.cache:
            return True
        else:
            try:
                self.load_to_cache(item)
            except KeyError:
                return False
            else:
                return True

    def new_id(self):
        id_ = self.newid
        self.newid += 1
        return id_

    def add_token(self, token):
        if token in self:
            self.get_value(token)[1] += 1
        else:
            self[token] = [self.new_id(), 1]

        self.requires_commit = True
        self.need_inv_index = True

    def load_to_cache(self, item):
        if isinstance(item, str):
            select, key, with_value = "id,count", "word", item
        elif isinstance(item, int):
            select, key, with_value = "word,count", "id", item

        try:
            value = list(next(iter(self.cur.execute(f"SELECT {select} FROM [vocabulary] WHERE {key} = ?", (with_value,)))))
        except StopIteration:
            raise KeyError()

        try:
            self.cache[item] = value
        except CacheFull:
            self.write_from_cache()
            self.cache.free()
        finally:
            self.cache[item] = value

        return value

    def get_value(self, item):
        if item not in self.cache:
            value = self.load_to_cache(item)
        else:
            value = self.cache[item]

        return value

    def __getitem__(self, item):
        return self.get_value(item)[0] # id or str

    def write_from_cache(self):
        for word, (id, count) in iter(self.cache):
            if isinstance(word, str):
                self.cur.execute("REPLACE INTO [vocabulary] (word, id, count) VALUES (?,?,?)", (word, id, count))
        self.commit()

    def get_count(self, id_):
        return self.get_value(id_)[1]

    def build_inv_index(self):
        self.cur.execute("CREATE INDEX IF NOT EXISTS id_index ON vocabulary(id)")
        self.commit()

    def __setitem__(self, key, value):
        try:
            self.cache[key] = value
        except CacheFull:
            self.write_from_cache()
            self.cache.free()

    def commit(self):
        self.db.commit()
        self.requires_commit = False

    def save(self, *args):
        self.write_from_cache()
        self.commit()

    def most_common(self, length=None):
        if length is None:
            length = len(self)
        self.write_from_cache()
        return [(id, word, count) for id, word, count in self.cur.execute(f"SELECT id, word, count FROM [vocabulary] LIMIT {length}").fetchall()]
        # return [(token_id, self.inv_ids[token_id], freq) for token_id, freq in self.count.most_common(length)]

    def __len__(self):
        self.write_from_cache()
        return self.cur.execute("SELECT COUNT() FROM [vocabulary]").fetchone()[0]

    def __del__(self):
        self.cur.close()
        self.db.close()



if __name__ == "__main__":
    import time
    st = time.time()
    print("Hello")
    # voc = Vocabulary()
    # voc = PersistentVocabulary(sys.argv[2], writeback=True)
    voc = SqliteVocabulary("d.db")

    import sys
    test_text = sys.argv[1]
    output_location = sys.argv[2]
    voc.add(open(test_text, "r", encoding="utf8").read().split())
    # voc.save(output_location)
    print(time.time()-st)
    voc.save("test_save")
    # for token_id, token, freq in voc.most_common():
    #     print("%d\ttoken_id, token, freq)

    # print("\n\n")
    # voc1 = Vocabulary.load("test_save")
    for token_id, token, freq in voc.most_common():
        print(token_id, token, freq)