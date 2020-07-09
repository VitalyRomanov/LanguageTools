#%%
from LanguageTools.DocumentCorpus import DocumentCorpus
from LanguageTools.rankers.BinaryRetrieval import BinaryRetrieval
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

    def __iter__(self):
        if self.file is not None:
            self.file.close()
        self.file = open(self.fname)
        return self

    def __next__(self):
        try:
            return json.loads(self.file.readline())["text"]
        except:
            raise StopIteration()

corpus = WikiReader(sys.argv[1])

#%%
print(os.getcwd())
#%%
dc = DocumentCorpus("test_doc_corpus", "ru")
dc.add_docs(corpus)
dc.save()

#%%
retr = BinaryRetrieval(dc)

#%%


