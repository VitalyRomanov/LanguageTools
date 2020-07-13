import os
import pickle as p
from collections import Counter
import mmap
from psutil import virtual_memory

from LanguageTools.DocumentCorpus import DocumentCorpus
from LanguageTools.rankers import SimilarityEngine
from LanguageTools.file_utils import check_dir_exists


def dump_shard(path, id, shard_postings):
    post_name = os.path.join(path, "temp_shard_posting_{0:06d}".format(id))
    indx_name = os.path.join(path, "temp_shard_index_{0:06d}".format(id))

    index = dict()

    with open(post_name, "wb") as shard_sink:
        for key, postings in shard_postings.items():
            position = shard_sink.tell()
            written = shard_sink.write(p.dumps(postings, protocol=4))

            index[key] = (position, written)

    p.dump(index, open(indx_name, "wb"), protocol=4)

    return (indx_name, post_name)


def merge_shards(path, vocab, shards):

    shards_ = []
    for index_path, shard_path in shards:
        shard_index = p.load(open(index_path, "rb"))
        f = open(shard_path, "r+b")
        mm = mmap.mmap(f.fileno(), 0)
        shards_.append((shard_index, mm))

    index = PostingIndex(path)

    for t_id, _, _ in vocab.most_common():
        p_ = set()

        for s_ind, s_file in shards_:
            if t_id in s_ind: # term never occurred in this shard
                pos_, len_ = s_ind[t_id]
                postings = p.loads(s_file[pos_: pos_+len_]) # shard store sets
                p_ |= postings

        index.add_posting(t_id, p_)

    index.save() # remove files

    for index_path, shard_path in shards:
        os.remove(index_path)
        os.remove(shard_path)

    return index




class BinaryRetrieval(SimilarityEngine):
    def __init__(self, corpus: DocumentCorpus, index_instantly=True):
        if not isinstance(corpus, DocumentCorpus):
            raise TypeError("courpus should be an instance of DocumentCorpus")

        super(BinaryRetrieval, self).__init__(corpus=corpus)

        self.inv_index = None
        if index_instantly:
            self.build_index() #move this out of constructor

    def build_index(self):
        # TODO
        # 1. include ngrams. alternative to ngrams is to use positional postings. however, ngram based implementation
        #   is simpler and allows to include misspell tolerant queries
        # 2. [v] (no need apparently) look into zarr
        # 3. [v] build disk-based index SPIMI see ref in 4.7
        # 4. compression (chapter 5)
        # 5. Correct orderings for ranked retrievals (chapter 6)
        # 6. Compare with SQLIte index

        shards = []
        shard_id = 0
        shard_postings = dict()



        for doc_ind, (id_, doc) in enumerate(self.corpus.corpus):
            for token in doc:
                assert token.id is not None, "token id is None, fatal error"
                if token.id not in shard_postings:
                    shard_postings[token.id] = set()
                shard_postings[token.id].add(id_)

            if doc_ind % 1000 == 0:
                size = virtual_memory().available / 1024 / 1024 #  total_size(shard_postings)
                # if size >= 1024*1024*1024: #1 GB
                if size <= 300:  # 100 MB
                    print(f"Only {size} mb of free RAM left")
                    shards.append(dump_shard(self.corpus.path, shard_id, shard_postings))

                    del shard_postings
                    shard_postings = dict()
                    shard_id += 1

            print(f"\rIndexing {doc_ind}/{len(self.corpus.corpus)}", end="")
        print(" " * 50, end="\r")

        if len(shard_postings) > 0:
            shards.append(dump_shard(self.corpus.path, shard_id, shard_postings))

        self.inv_index = merge_shards(self.corpus.path, self.corpus.corpus.vocab, shards)

    def retrieve_sub_rank(self, tokens):

        doc_ranks = Counter()

        for t_id, rank in tokens:
            for doc_id in self.inv_index[t_id]:
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = 0.
                doc_ranks[doc_id] += rank

        return doc_ranks

    def retrieve_sub(self, tokens):
        # this does not work with query expansion

        doc_ranks = Counter()
        for doc_id in set.intersection(*[self.inv_index[t_id] for t_id, rank in tokens]):
            doc_ranks[doc_id] = 1.

        return doc_ranks

    def save(self, path):
        check_dir_exists(path)

        # p.dump(self.inv_index, open(os.path.join(path, "inv_index"), "wb"))
        p.dump(self.corpus.path, open(os.path.join(path, "corpus_ref"), "wb"))
        p.dump(self.corpus.path, open(os.path.join(path, "postings_ref"), "wb"))

    @classmethod
    def load(cls, path):
        corpus = DocumentCorpus.load(p.load(open(os.path.join(path, "corpus_ref"), "rb")))
        inv_index = PostingIndex.load(p.load(open(os.path.join(path, "postings_ref"), "rb")))

        retr = BinaryRetrieval(corpus, index_instantly=False)
        retr.inv_index = inv_index
        # retr.inv_index = p.load(open(os.path.join(path, "inv_index"), "rb"))
        return retr


class PostingIndex:
    file_index = None
    index = None

    def __init__(self, path, shard_size=2**30):

        self.file_index = dict() # (shard, filename)
        self.index = dict() # (shard, position, length)

        self.opened_shards = dict() # (shard, file, mmap object) if mmap is none -> opened for write

        self.shard_for_write = 0
        self.written_in_current_shard = 0
        self.shard_size=shard_size

        self.path = path

    def add_posting(self, term_id, postings):

        serialized_doc = p.dumps(postings, protocol=4)

        f, _ = self.writing_mode(self.shard_for_write)

        position = f.tell()
        written = f.write(serialized_doc)
        self.index[term_id] = (self.shard_for_write, position, written)
        self.increment_byte_count(written)
        return term_id

    def increment_byte_count(self, written):
        self.written_in_current_shard += written
        if self.written_in_current_shard >= self.shard_size:
            self.shard_for_write += 1
            self.written_in_current_shard = 0

    def __getitem__(self, doc_id):
        return self.get_with_id(doc_id)

    # def __iter__(self):
    #     self.iter_doc = 0
    #     return self
    #
    # def __next__(self):
    #     if self.iter_doc >= len(self.index):
    #         raise StopIteration()
    #     c_doc_id = self.iter_doc
    #     self.iter_doc += 1
    #     return c_doc_id, self[c_doc_id]

    def get_with_id(self, doc_id):
        shard, pos, len_ = self.index[doc_id]
        _, mm = self.reading_mode(shard)
        return p.loads(mm[pos: pos+len_])

    def get_name_format(self, id_):
        return 'postings_shard_{0:04d}'.format(id_)

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
            self.file_index,
            self.shard_for_write,
            self.written_in_current_shard,
            self.shard_size,
            self.path
        ), open(os.path.join(self.path, "postings_params"), "wb"), protocol=4)

    def load_param(self):
        self.file_index,\
        self.shard_for_write,\
        self.written_in_current_shard,\
        self.shard_size,\
        self.path = p.load(open(os.path.join(self.path, "postings_params"), "rb"))

    def save_index(self):
        p.dump(self.index, open(os.path.join(self.path, "postings_index"), "wb"), protocol=4)

    def load_index(self):
        self.index = p.load(open(os.path.join(self.path, "postings_index"), "rb"))

    def save(self):
        self.save_index()
        self.save_param()

    @classmethod
    def load(cls, path):
        postings = PostingIndex(path)
        postings.load_param()
        postings.load_index()
        return postings


    def close_all_shards(self):
        for shard in self.opened_shards.values():
            for s in shard:
                if s:
                    s.close()