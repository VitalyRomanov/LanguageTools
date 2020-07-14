import os
import numpy as np

def check_dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# TODO
class CompactStorage:
    def __init__(self, n_fields=1, init_size=100000, dtype=np.uint32):
        if n_fields == 0:
            raise ValueError("The parameter n_fields should be greater than 0")

        if n_fields == 1:
            self.storage = np.zeros(shape=(init_size, ), dtype=dtype)
        else:
            self.storage = np.zeros(shape=(init_size, n_fields), dtype=dtype)

        self.active_storage_size = 0

    def resize_storage(self, new_size):
        if len(self.storage.shape) == 1:
            self.storage.resize((new_size,))
        else:
            self.storage.resize((new_size,self.storage.shape[1]))

    def __len__(self):
        return self.active_storage_size

    def __getitem__(self, item):
        if item >= self.active_storage_size:
            raise IndexError("Out of range:", item)

        if len(self.storage.shape) == 1:
            return self.storage[item]
        else:
            return tuple(self.storage[item])

    def __setitem__(self, key, value):
        # if key >= len(self):
        #     self.resize_storage(int(self.storage.shape[0]*1.2))

        if key >= self.active_storage_size:
            raise IndexError("Out of range:", key)

        if len(self.storage.shape) == 1:
            self.storage[key] = value
        else:
            self.storage[key,:] = np.fromiter(value, dtype=self.storage.dtype)

    def append(self, value):
        if self.active_storage_size >= self.storage.shape[0]:
            self.resize_storage(int(self.storage.shape[0]*1.2))

        if len(self.storage.shape) == 1:
            self.storage[self.active_storage_size] = value
        else:
            self.storage[self.active_storage_size,:] = np.fromiter(value, dtype=self.storage.dtype)

        self.active_storage_size += 1

