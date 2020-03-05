from collections import Counter
from numpy.random import choice
from numpy import array
import numpy as np
import pickle as p
from time import time
import mmh3

class Vocabulary:

    def __init__(self):
        self.count = Counter()
        self.ids = {}
        self.inv_ids = {}
        self.prob_valid = False

        self.add(['UNK'])


    def add(self, tokens):
        for token in tokens:
            if token in self.ids:
                self.count[self.ids[token]] += 1
            else:
                new_id = len(self.ids)
                self.ids[token] = new_id
                self.inv_ids[new_id] = token
                self.count[new_id] = 1

        self.prob_valid = False

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
        # p.dump(self, open(destination, "wb"))
        
        with open(destination, "w", encoding="utf8") as voc_sink:
            voc_sink.write("Version: %d\n" % time())
            for token, token_id in self.ids.items():
                voc_sink.write("%d\t%s\t%d\n" % (token_id, token, self.count[token_id]))



    @classmethod
    def load(self, location):

        voc = Vocabulary()
        with open(location, "r", encoding="utf8") as voc_source:
            for ind, line in enumerate(voc_source):
                if ind == 0 or ind == 1: continue
                if line.strip():
                    token_id, token, freq = line.strip().split()
                    voc.ids[token] = token_id
                    voc.inv_ids[token_id] = token
                    voc.count[token_id] = int(freq)

        voc.calculate_prob()
        return voc
        # return p.load(open(location, "rb"))


if __name__ == "__main__":
    voc = Vocabulary()

    import sys
    test_text = sys.argv[1]
    output_location = sys.argv[2]
    voc.add(open(test_text, "r", encoding="utf8").read().split())
    voc.save(output_location)
    # voc.save("test_save")
    # for token_id, token, freq in voc.most_common():
    #     print("%d\ttoken_id, token, freq)

    # print("\n\n")
    # voc1 = Vocabulary.load("test_save")
    # for token_id, token, freq in voc.most_common():
    #     print(token_id, token, freq)