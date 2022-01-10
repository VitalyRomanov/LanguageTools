import json
import logging
import os
import pickle
from ast import literal_eval
from collections import defaultdict, deque
from itertools import groupby, chain
from mmap import mmap, ACCESS_READ
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

from psutil import Process, virtual_memory
from tqdm import tqdm


def check_dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# class MapReduceWriter:
#     @staticmethod
#     def serialize(obj):
#         return json.dumps(pickle.dumps(obj, protocol=0).decode("ascii"))
#
#     @staticmethod
#     def deserialize(line):
#         return pickle.loads(json.loads(line).encode("ascii"))
#
#     @classmethod
#     def kv_to_str(cls, key, val):
#         return f"{repr(key)} {cls.serialize(val)}"
#
#     @staticmethod
#     def get_key_from_record(line):
#         return literal_eval(line[:line.find(" ")])
#
#     @classmethod
#     def get_value_from_records(cls, line):
#         return cls.deserialize(line[line.find(" ") + 1:])
#
#     @classmethod
#     def get_kv_from_record(cls, line):
#         pos = line.find(" ")
#         return literal_eval(line[:pos]), cls.deserialize(line[pos + 1:])


class Shuffler:
    delim = "\t"

    def __init__(self, path: Path):
        self.path = path
        self.parts = []
        self.currently_written = 0
        self.opened_file = None

        self.open_next_file()

    @staticmethod
    def serialize(obj):
        return json.dumps(pickle.dumps(obj, protocol=0).decode("ascii"))

    @staticmethod
    def deserialize(line):
        return pickle.loads(json.loads(line).encode("ascii"))

    @classmethod
    def kv_to_str(cls, key, val):
        if isinstance(key, str):
            assert cls.delim not in key
        return f"{repr(key)}{cls.delim}{cls.serialize(val)}"

    @classmethod
    def get_key_from_record(cls, line):
        return literal_eval(line[:line.find(cls.delim)])

    @classmethod
    def get_value_from_records(cls, line):
        return cls.deserialize(line[line.find(cls.delim) + 1:])

    @classmethod
    def get_kv_from_record(cls, line):
        pos = line.find(cls.delim)
        return literal_eval(line[:pos]), cls.deserialize(line[pos + 1:])

    def close_last_file(self):
        if self.opened_file is not None and self.opened_file.closed is False:
            self.opened_file.close()

    def open_next_file(self):
        ind = len(self.parts)
        file_path = self.path.joinpath(f"part_{ind}")
        self.parts.append(file_path)

        self.close_last_file()
        self.opened_file = open(file_path, "a")

    def write(self, key, value, bytes_per_file=104857600):
        to_write = self.kv_to_str(key, value)
        self.opened_file.write(f"{to_write}\n")
        self.currently_written += len(to_write)

        if self.currently_written > bytes_per_file:
            self.open_next_file()
            self.currently_written = 0

    def sort_file(self, path):
        with open(path, "r") as source:
            lines = source.readlines()

        # with Pool(4) as p:
        keys = map(self.get_key_from_record, lines)

        records = list(zip(keys, lines))

        records.sort(key=lambda record: record[0])
        return records

    def merge_sorted(self, first, second, output):
        sorted_position = 0
        first_line = first.readline()
        second_line = second[sorted_position]

        with open(output, "w") as sink:
            while first_line != "" or sorted_position < len(second):
                if first_line != "" and (second_line is None or self.get_key_from_record(first_line) <= second_line[0]):
                    sink.write(first_line)
                    first_line = first.readline()
                else:
                    sink.write(second_line[1])
                    sorted_position += 1
                    if sorted_position < len(second):
                        second_line = second[sorted_position]
                    else:
                        second_line = None

                # if first_line == "":
                #     sink.write(second_line[1])
                #     sorted_position += 1
                # else:
                #     if self.get_key_from_record(first_line) <= second_line[0]:
                #         sink.write(first_line)
                #     else:
                #         sink.write(second_line[1])
                #         sorted_position += 1



    def sort_first_file(self, file):
        sorted_path = file.parent.joinpath(f"{file.name}_sorted")

        with open(sorted_path, "w") as sink:
            sorted_lines = self.sort_file(file)
            for key_line in sorted_lines:
                sink.write(key_line[1])

        return sorted_path

    def sort(self):
        assert len(self.parts) > 0

        last_sorted = self.sort_first_file(self.parts.pop(0))

        for ind, file in enumerate(tqdm(self.parts, desc="Sorting...")):
            sorted_path = file.parent.joinpath(f"{file.name}_sorted")

            second = self.sort_file(file)

            with open(last_sorted, "r") as first:
                self.merge_sorted(first, second, sorted_path)

            if last_sorted is not None:
                os.remove(last_sorted)
            last_sorted = sorted_path

        return last_sorted

    def get_sorted(self):
        self.close_last_file()

        sorted_path = self.sort()
        with open(sorted_path, "r") as sorted_:
            for line in sorted_:
                k, v = self.get_kv_from_record(line.strip())
                yield k, v


class MapReduce:
    """
    Performs a local Map Reduce job. The goal of this class is to process large files that do not fit into memory.
    """
    def __init__(self, path: Path, map_fn, reduce_fn):
        """
        :param path: Path where intermediate results are stored
        :param map_fn:
        :param reduce_fn:
        :param num_partition_buckets: limited by the number of simultaneously opened files in your os
        """
        self.path = path.joinpath("shard_partitions")
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
        if self.path.is_dir():
            raise Exception(f"Temp directory exists: {self.path}")
        self.path.mkdir()
        # self.files = dict()
        self.shuffler = Shuffler(self.path)

    # def get_partition(self, key):
    #     return key % self.num_files

    # def get_file(self, id_):
    #     if id_ not in self.files:
    #         self.files[id_] = Shuffler(self.path.joinpath(f"shard_{id_}"))
    #         # self.files[id_] = open(self.path.joinpath(f"shard_{id_}"), "a")
    #     return self.files[id_]

    def dump_shard(self, shard):
        for key, val in shard.items():
            self.shuffler.write(key, val)
            # f = self.get_file(self.get_partition(key))
            # f.write(f"{repr(key)} {self.serialize(val)}\n")

    # def get_keys_offsets(self, file, batch_size=1000000):
    #     need_processing = True
    #     processed = set()
    #     pending = set()
    #
    #     total_lines = None
    #     line_count = 0
    #
    #     while need_processing:
    #         keys_offsets = []
    #         start = 0
    #         end = 0
    #
    #         file.seek(0)
    #         line = file.readline()
    #         while line != b"":
    #             line_ind = line_count
    #             line_count += 1
    #
    #             ln = len(line)  # only ascii
    #             end += ln
    #             key = self.get_key_from_record(line)
    #             add_now = False
    #
    #             if line_ind not in processed:
    #                 if key not in pending and len(pending) < batch_size:
    #                     processed.add(line_ind)
    #                     pending.add(key)
    #                     add_now = True
    #                 elif key not in pending and len(pending) >= batch_size:
    #                     pass
    #                 else:
    #                     processed.add(line_ind)
    #                     add_now = True
    #
    #             if add_now:
    #                 keys_offsets.append((key, start, end))
    #             start += ln
    #             line = file.readline()
    #
    #         if len(keys_offsets) > 0:
    #             if total_lines is None:
    #                 total_lines = line_count
    #             line_count = 0
    #             yield keys_offsets
    #         else:
    #             need_processing = len(processed) < total_lines
    #
    #         keys_offsets.clear()
    #         pending.clear()

    # def reduce_lists(self, key, offsets, mm):
    #     value = None
    #     for key, start, end in offsets:
    #         value = self.reduce_fn(value, self.get_value_from_records(mm[start:end]))
    #     return value

    # def merge(self, file, file_ind, total_files):
    #     with open(file, "rb") as partition:
    #         mm = mmap(partition.fileno(), 0, access=ACCESS_READ)
    #         for keys_offsets in self.get_keys_offsets(mm):
    #             for key, offsets in groupby(keys_offsets, key=lambda x: x[0]):
    #                 yield key, self.reduce_lists(key, offsets, mm)
    #
    # def iterate_merged(self):
    #     files = list(self.path.iterdir())
    #     for ind, file in enumerate(files):
    #         if not file.name.startswith("shard_"):
    #             continue
    #         for key, val in self.merge(file, ind, len(files)):
    #             yield key, val

    @staticmethod
    def create_buffer_storage():
        return defaultdict(lambda: None)

    def run(self, data, allow_parallel=False, total=None, desc="", spill_key_buffer_threshold=2000000):

        temp_storage_shard = self.create_buffer_storage()
        batch_size = 1000
        buffer = deque()

        for ind, id_val in tqdm(
                enumerate(data), total=total, desc=desc
        ):
            buffer.append(id_val)

            if len(buffer) >= batch_size:
                if allow_parallel:
                    with Pool(4) as p:
                        mapped = p.map(self.map_fn, buffer)
                else:
                    mapped = map(self.map_fn, buffer)

                for map_id, map_val in chain(*mapped):
                    temp_storage_shard[map_id] = self.reduce_fn(temp_storage_shard[map_id], map_val)

                # if len(temp_storage_shard) >= spill_key_buffer_threshold:
                size = Process(os.getpid()).memory_info().rss / 1024 / 1024  # memory usage is MB
                free_size = virtual_memory().available / 1024 / 1024  # total_size(postings_shard)
                if size >= 4000 or free_size < 500:
                    # print(f"Only {size} mb of free RAM left")
                    # print(f"Using {size} mb of RAM, {free_size} mb free")
                    self.dump_shard(temp_storage_shard)

                    del temp_storage_shard
                    temp_storage_shard = self.create_buffer_storage()

                buffer.clear()

        if len(temp_storage_shard) > 0:
            self.dump_shard(temp_storage_shard)

        last_key = None
        reduced_value = None
        for key, value in self.shuffler.get_sorted():
            if last_key != key and last_key is not None:
                yield last_key, reduced_value
                reduced_value = None

            reduced_value = self.reduce_fn(reduced_value, value)
            last_key = key

        if last_key is not None:
            yield last_key, reduced_value

    def __del__(self):
        # for file in self.files.values():
        #     if file is not None and file.closed is False:
        #         file.close()
        rmtree(self.path)


# class CompactStorage:
#     def __init__(self, n_fields=1, init_size=100000, dtype=np.uint32, volatile_access=False):
#         if n_fields == 0:
#             raise ValueError("The parameter n_fields should be greater than 0")
#
#         self.volatile_access = volatile_access
#
#         if n_fields == 1:
#             self.storage = np.zeros(shape=(init_size, ), dtype=dtype)
#         else:
#             self.storage = np.zeros(shape=(init_size, n_fields), dtype=dtype)
#
#         self.active_storage_size = 0
#
#     def resize_storage(self, new_size):
#         if len(self.storage.shape) == 1:
#             self.storage.resize((new_size,))
#         else:
#             self.storage.resize((new_size,self.storage.shape[1]))
#
#     def __len__(self):
#         if self.volatile_access:
#             return self.storage.shape[0]
#         else:
#             return self.active_storage_size
#
#     def __getitem__(self, item):
#         if item >= len(self):
#             raise IndexError("Out of range:", item)
#
#         if len(self.storage.shape) == 1:
#             return self.storage[item]
#         else:
#             return tuple(self.storage[item])
#
#     def __setitem__(self, key, value):
#         # if self.volatile_access:
#         #     if key >= len(self):
#         #         self.resize_storage(int(key*1.2))
#
#         if key >= len(self):
#             # if self.volatile_access:
#             #     raise IndexError("Out of range:", key, "Preallocate when using volatile_access=True")
#             raise IndexError("Out of range:", key)
#
#         if len(self.storage.shape) == 1:
#             self.storage[key] = value
#         else:
#             self.storage[key,:] = np.fromiter(value, dtype=self.storage.dtype)
#
#     def append(self, value):
#         if self.volatile_access:
#             raise Exception("Use __setitem__ when volatile_access=True")
#
#         if self.active_storage_size >= self.storage.shape[0]:
#             self.resize_storage(int(self.storage.shape[0]*1.2))
#
#         if len(self.storage.shape) == 1:
#             self.storage[self.active_storage_size] = value
#         else:
#             self.storage[self.active_storage_size,:] = np.fromiter(value, dtype=self.storage.dtype)
#
#         self.active_storage_size += 1
#
#
# class DbDict:
#     def __init__(self, path, keytype=str):
#         self.path = path
#         self.conn = sqlite3.connect(path)
#         self.cur = self.conn.cursor()
#         self.requires_commit = False
#
#         if keytype == str:
#             keyt_ = "VARCHAR(255)"
#         elif keytype == int:
#             keyt_ = "INTEGER"
#         else:
#             raise ValueError("Keytype only supports str and int")
#
#         self.cur.execute("CREATE TABLE IF NOT EXISTS [mydict] ("
#                          "[key] %s PRIMARY KEY NOT NULL, "
#                          "[value] BLOB)" % keyt_)
#
#     def __setitem__(self, key, value):
#         val = sqlite3.Binary(pickle.dumps(value, protocol=4))
#         self.cur.execute("REPLACE INTO [mydict] (key, value) VALUES (?, ?)",
#                          (key, val))
#
#         self.requires_commit = True
#
#
#     def __getitem__(self, key):
#         if self.requires_commit:
#             self.commit()
#             self.requires_commit = False
#
#         try:
#             val = next(iter(self.conn.execute("SELECT value FROM [mydict] WHERE key = ?", (key,))))[0]
#         except StopIteration:
#             raise KeyError("Key not found")
#
#         return pickle.loads(bytes(val))
#
#     def get(self, item, default):
#         try:
#             return self[item]
#         except KeyError:
#             return default
#
#     def __delitem__(self, key):
#         try:
#             self.conn.execute("DELETE FROM [mydict] WHERE key = ?", (key,))
#         except:
#             pass
#
#     def commit(self):
#         self.conn.commit()
#
#     def close(self):
#         self.cur.close()
#         self.conn.close()
#
#
# class CompactKeyValueStore:
#     file_index = None
#     index = None
#     key_map = None
#
#     def __init__(self, path, init_size=100000, shard_size=2**30, compact_ensured=False):
#
#         self.path = path
#
#         self.file_index = dict() # (shard, filename)
#
#
#         if not compact_ensured:
#             self.key_map = dict()
#             self.index = CompactStorage(3, init_size)
#         else:
#             self.index = CompactStorage(3, init_size, volatile_access=True)
#
#         self.opened_shards = dict() # (shard, file, mmap object) if mmap is none -> opened for write
#
#         self.shard_for_write = 0
#         self.written_in_current_shard = 0
#         self.shard_size=shard_size
#
#     def init_storage(self, size):
#         self.index.active_storage_size = size
#
#     def __setitem__(self, key, value, key_error='ignore'):
#         if self.key_map is not None:
#             if key not in self.key_map:
#                 self.key_map[key] = len(self.index)
#             key_: int = self.key_map[key]
#         else:
#             if not isinstance(key, int):
#                 raise ValueError("Keys should be integers when setting compact_ensured=True")
#             key_: int = key
#
#         serialized_doc = pickle.dumps(value, protocol=4)
#
#         try:
#             existing_shard, existing_pos, existing_len = self.index[key_] # check if there is an entry with such key
#         except IndexError:
#             pass
#         else:
#             if len(serialized_doc) == existing_len:
#                 # successfully retrieved existing position and can overwrite old data
#                 _, mm = self.reading_mode(existing_shard)
#                 mm[existing_pos: existing_pos + existing_len] = serialized_doc
#                 return
#
#         # no old data or the key is new
#         f, _ = self.writing_mode(self.shard_for_write)
#         position = f.tell()
#         written = f.write(serialized_doc)
#         if self.index.volatile_access: # no key_map available, index directly by key
#             if key_ >= len(self.index):
#                 # make sure key densely follow each other, otherwise a lot
#                 # of space is wasted
#                 self.index.resize_storage(int(key_ * 1.2))
#             # access with key_ directly
#             self.index[key_] = (self.shard_for_write, position, written)
#         else:
#             # append, index length is maintained by self.index itself
#             self.index.append((self.shard_for_write, position, written))
#         self.increment_byte_count(written)
#
#     # def add_posting(self, term_id, postings):
#     #     if self.index is None:
#     #         raise Exception("Index is not initialized")
#     #
#     #     serialized_doc = pickle.dumps(postings, protocol=4)
#     #
#     #     f, _ = self.writing_mode(self.shard_for_write)
#     #
#     #     position = f.tell()
#     #     written = f.write(serialized_doc)
#     #     self.index[term_id] = (self.shard_for_write, position, written)
#     #     self.increment_byte_count(written)
#     #     return term_id
#
#     def increment_byte_count(self, written):
#         self.written_in_current_shard += written
#         if self.written_in_current_shard >= self.shard_size:
#             self.shard_for_write += 1
#             self.written_in_current_shard = 0
#
#     def __getitem__(self, key):
#         if self.key_map is not None:
#             if key not in self.key_map:
#                 raise KeyError("Key does not exist:", key)
#             key_ = self.key_map[key]
#         else:
#             key_ = key
#             if key_ >= len(self.index):
#                 raise KeyError("Key does not exist:", key)
#         try:
#             return self.get_with_id(key_)
#         except ValueError:
#             raise KeyError("Key does not exist:", key)
#
#     def get_with_id(self, doc_id):
#         shard, pos, len_ = self.index[doc_id]
#         if len_ == 0:
#             raise ValueError("Entry lenght is 0")
#         _, mm = self.reading_mode(shard)
#         return pickle.loads(mm[pos: pos+len_])
#
#     def get_name_format(self, id_):
#         return 'postings_shard_{0:04d}'.format(id_)
#
#     def open_for_read(self, name):
#         # raise filenotexists
#         f = open(os.path.join(self.path, name), "r+b")
#         mm = mmap.mmap(f.fileno(), 0)
#         return f, mm
#
#     def open_for_write(self, name):
#         # raise filenotexists
#         self.check_dir_exists()
#         f = open(os.path.join(self.path, name), "ab")
#         return f, None
#
#     def check_dir_exists(self):
#         if not os.path.isdir(self.path):
#             os.mkdir(self.path)
#
#     def writing_mode(self, id_):
#         if id_ not in self.opened_shards:
#             if id_ not in self.file_index:
#                 self.file_index[id_] = self.get_name_format(id_)
#             self.opened_shards[id_] = self.open_for_write(self.file_index[id_])
#         elif self.opened_shards[id_][1] is not None: # mmap is None
#             self.opened_shards[id_][1].close()
#             self.opened_shards[id_][0].close()
#             self.opened_shards[id_] = self.open_for_write(self.file_index[id_])
#         return self.opened_shards[id_]
#
#     def reading_mode(self, id_):
#         if id_ not in self.opened_shards:
#             if id_ not in self.file_index:
#                 self.file_index[id_] = self.get_name_format(id_)
#             self.opened_shards[id_] = self.open_for_read(self.file_index[id_])
#         elif self.opened_shards[id_][1] is None:
#             self.opened_shards[id_][0].close()
#             self.opened_shards[id_] = self.open_for_read(self.file_index[id_])
#         return self.opened_shards[id_]
#
#     def save_param(self):
#         pickle.dump((
#             self.file_index,
#             self.shard_for_write,
#             self.written_in_current_shard,
#             self.shard_size,
#             self.path
#         ), open(os.path.join(self.path, "postings_params"), "wb"), protocol=4)
#
#     def load_param(self):
#         self.file_index,\
#         self.shard_for_write,\
#         self.written_in_current_shard,\
#         self.shard_size,\
#         self.path = pickle.load(open(os.path.join(self.path, "postings_params"), "rb"))
#
#     def save_index(self):
#         pickle.dump(self.index, open(os.path.join(self.path, "postings_index"), "wb"), protocol=4)
#
#     def load_index(self):
#         self.index = pickle.load(open(os.path.join(self.path, "postings_index"), "rb"))
#
#     def save(self):
#         self.save_index()
#         self.save_param()
#         self.close_all_shards()
#
#     @classmethod
#     def load(cls, path):
#         postings = CompactKeyValueStore(path)
#         postings.load_param()
#         postings.load_index()
#         return postings
#
#
#     def close_all_shards(self):
#         for shard in self.opened_shards.values():
#             for s in shard[::-1]:
#                 if s:
#                     s.close()
#
#     def close(self):
#         self.close_all_shards()
#
#     def commit(self):
#         for shard in self.opened_shards.values():
#             if shard[1] is not None:
#                 shard[1].flush()