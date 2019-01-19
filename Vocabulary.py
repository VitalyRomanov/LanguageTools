from collections import Counter
from numpy.random import choice
from numpy import array
import numpy as np

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

    def text2ids(self, tokens):
        return [self.ids[token] for token in tokens]

    def ids2text(self, ids):
        return [self.inv_ids[id] for id in ids]

    def __len__(self):
        return len(self.count)

    def most_common(self, length=None):
        if length is None:
            length = len(self)
        return self.count.most_common(length)

    def values(self):
        return self.count.values()

    def calculate_prob(self):
        p = array(list(self.count.values()))
        self.p = p / np.sum(p)
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
