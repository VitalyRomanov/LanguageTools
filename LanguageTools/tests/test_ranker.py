#%%
from LanguageTools.corpus import DocumentCorpus
from LanguageTools.rankers import BinaryRetrieval
import os, json
import sys

#%%
# corpus = None
# with open("/Volumes/External/dev/LanguageTools/LanguageTools/test_data/test_corpus.txt") as tcorpus:
#     corpus = tcorpus.read().strip().split("\n\n")

class WikiReader:
    def __init__(self, fname):
        self.fname = fname
        self.file = None
        self.cached_len = None

        len(self)

    def __len__(self):
        if self.cached_len is None:
            self.cached_len = 0
            with open(self.fname) as file:
                for _ in file.readline():
                    self.cached_len += 1
        return self.cached_len

    def __iter__(self):
        if self.file is not None:
            self.file.close()
        self.file = open(self.fname)
        self.cindex = 0
        return self

    def __next__(self):
        try:
            print(f"\r{self.cindex}/{len(self)}", end="")
            self.cindex += 1
            return json.loads(self.file.readline())["text"]
        except:
            raise StopIteration()

corpus = WikiReader("/home/ltv/data/local_run/wikipedia/wiki_en.joined")
# corpus = WikiReader("/Volumes/External/datasets/Language/Corpus/en/EnWiki/en_wiki_100mb.txt")

#%%
print(os.getcwd())
#%%
# dc = DocumentCorpus("test_doc_corpus_en", "en")
# dc.add_docs(corpus, save_instantly=False)
# dc.save()

dc = DocumentCorpus.load("test_doc_corpus_en")
#
# %%
retr = BinaryRetrieval(dc)
retr.save()
# retr = BinaryRetrieval.load("binary_retr_en")


#%%
retr.query("political philosophy", exact=False)

#%%
