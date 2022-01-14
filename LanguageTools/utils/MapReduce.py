import json
import os
import pickle
from ast import literal_eval
from bisect import bisect_left
from collections import defaultdict, deque
from copy import copy
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

from psutil import Process, virtual_memory
from tqdm import tqdm


class Shuffler:
    delim = "\t"

    def __init__(self, path: Path):
        self.path = path
        self.parts = []
        self.currently_written = 0
        self.opened_file = None

        # self.open_next_file()

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

    # def write(self, key, value, bytes_per_file=104857600):  # 104857600
    #     to_write = self.kv_to_str(key, value)
    #     self.opened_file.write(f"{to_write}\n")
    #     self.currently_written += len(to_write)
    #
    #     if self.currently_written > bytes_per_file:
    #         self.open_next_file()
    #         self.currently_written = 0

    def write_sorted(self, shard):
        keys = list(shard.keys())
        keys.sort()

        self.open_next_file()

        for key in keys:
            value = shard[key]
            to_write = self.kv_to_str(key, value)
            self.opened_file.write(f"{to_write}\n")



    # def sort_file(self, path):
    #     with open(path, "r") as source:
    #         lines = source.readlines()
    #
    #     # with Pool(4) as p:
    #     keys = map(self.get_key_from_record, lines)
    #
    #     records = list(zip(keys, lines))
    #
    #     records.sort(key=lambda record: record[0])
    #     return records

    # def merge_sorted(self, first, second, output):
    #     sorted_position = 0
    #     first_line = first.readline()
    #     second_line = second[sorted_position]
    #
    #     with open(output, "w") as sink:
    #         while first_line != "" or sorted_position < len(second):
    #             if first_line != "" and (second_line is None or self.get_key_from_record(first_line) <= second_line[0]):
    #                 sink.write(first_line)
    #                 first_line = first.readline()
    #             else:
    #                 sink.write(second_line[1])
    #                 sorted_position += 1
    #                 if sorted_position < len(second):
    #                     second_line = second[sorted_position]
    #                 else:
    #                     second_line = None
    #
    # def sort_first_file(self, file):
    #     sorted_path = file.parent.joinpath(f"{file.name}_sorted")
    #
    #     with open(sorted_path, "w") as sink:
    #         sorted_lines = self.sort_file(file)
    #         for key_line in sorted_lines:
    #             sink.write(key_line[1])
    #
    #     return sorted_path

    @staticmethod
    def close_empty(files, first_lines):
        filtered_files = []
        filtered_lines = []

        for file, line in zip(files, first_lines):
            if line == "":
                file.close()
            else:
                filtered_files.append(file)
                filtered_lines.append(line)

        return filtered_files, filtered_lines

    def merge_files(self, files, level, ord):
        f = [open(file, "r") for file in files]
        l = [file.readline() for file in f]

        # f, l = self.close_empty(f, l)

        keys = [self.get_key_from_record(line) for line in l]

        sorted_ind = sorted(list(range(len(keys))), key=lambda x: keys[x])
        sorted_keys = [keys[i] for i in sorted_ind]
        sorted_lines = [l[i] for i in sorted_ind]

        del keys
        del l

        sorted_path = self.path.joinpath(f"sorted_{level}_{ord}")

        with open(sorted_path, "w") as sink:
            while len(sorted_ind) > 0:
                min_ind = sorted_ind.pop(0)
                _ = sorted_keys.pop(0)
                min_line = sorted_lines.pop(0)
                sink.write(min_line)

                new_line = f[min_ind].readline()

                if new_line == "":
                    f[min_ind].close()
                    # f.pop(min_ind)
                else:
                    new_key = self.get_key_from_record(new_line)
                    ins_pos = bisect_left(sorted_keys, new_key)

                    sorted_lines.insert(ins_pos, new_line)
                    sorted_keys.insert(ins_pos, new_key)
                    sorted_ind.insert(ins_pos, min_ind)

        # min_key = keys[0]
        # min_key_ind = 0
        #
        # sorted_path = self.path.joinpath(f"sorted_{level}_{ord}")
        #
        # with open(sorted_path, "w") as sink:
        #     while True:
        #         for ind, key in enumerate(keys):
        #             if key is not None and key < min_key:
        #                 min_key = key
        #                 min_key_ind = ind
        #
        #         if l[min_key_ind] == "":
        #             break
        #
        #         sink.write(l[min_key_ind])
        #         l[min_key_ind] = f[min_key_ind].readline()
        #         cline = l[min_key_ind]
        #         keys[min_key_ind] = self.get_key_from_record(cline) if cline != "" else None

        return sorted_path

    def merge_sorted(self, files, merge_chunks=20, level=0):

        if len(files) == 1:
            return files[0]

        to_merge = []
        merged = []

        while len(files) > 0:
            to_merge.append(files.pop(0))

            if len(to_merge) >= merge_chunks or len(files) == 0:
                merged.append(self.merge_files(to_merge, level, ord=len(merged)))
                to_merge.clear()

        return self.merge_sorted(merged, level=level+1)

    def sort(self):
        assert len(self.parts) > 0

        return self.merge_sorted(self.parts)

    # def sort(self):
    #     assert len(self.parts) > 0
    #
    #     last_sorted = self.sort_first_file(self.parts.pop(0))
    #
    #     for ind, file in enumerate(tqdm(self.parts, desc="Sorting...")):
    #         sorted_path = file.parent.joinpath(f"{file.name}_sorted")
    #
    #         second = self.sort_file(file)
    #
    #         with open(last_sorted, "r") as first:
    #             self.merge_sorted(first, second, sorted_path)
    #
    #         if last_sorted is not None:
    #             os.remove(last_sorted)
    #         last_sorted = sorted_path
    #
    #     return last_sorted

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
        """
        self.path = path.joinpath("map_reduce_job")
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
        if self.path.is_dir():
            raise Exception(f"Temp directory exists: {self.path}")
        self.path.mkdir()
        # self.files = dict()
        self.shuffler = Shuffler(self.path)

    def dump_shard(self, shard):
        self.shuffler.write_sorted(shard)

    @staticmethod
    def create_buffer_storage():
        return defaultdict(lambda: None)

    def process_batch(self, buffer, temp_storage_shard, allow_parallel):
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

        return temp_storage_shard

    def run(self, data, allow_parallel=False, total=None, desc="", spill_key_buffer_threshold=2000000):

        temp_storage_shard = self.create_buffer_storage()
        batch_size = 10000
        buffer = deque()

        for ind, id_val in tqdm(
                enumerate(data), total=total, desc=desc
        ):
            buffer.append(id_val)

            if len(buffer) >= batch_size:
                temp_storage_shard = self.process_batch(buffer, temp_storage_shard, allow_parallel)
                buffer.clear()

        if len(buffer) > 0:
            temp_storage_shard = self.process_batch(buffer, temp_storage_shard, allow_parallel)

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
        rmtree(self.path)