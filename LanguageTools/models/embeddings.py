import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np


Embeddings = namedtuple("Embeddings", ["mapping", "vectors"])


def load_w2v(path, load_n_vectors=1000000):
    path = Path(path)
    cached_path = path.parent.joinpath(path.name + "___cached")
    if not cached_path.is_file():
        id_map = {}
        vecs = []
        with open(path, encoding="UTF-8", errors='ignore') as vectors:
            n_vectors, n_dims = map(int, vectors.readline().strip().split())
            n_vectors = min(n_vectors, load_n_vectors)

            for ind in range(n_vectors):
                elements = vectors.readline().strip().split()

                vec = list(map(float, elements[1:]))
                assert len(vec) == n_dims
                id_map[elements[0]] = len(vecs)
                vecs.append(vec)
        vecs = np.array(vecs)
        pickle.dump((id_map, vecs), open(cached_path, "wb"))
    else:
        id_map, vecs = pickle.load(open(cached_path, "rb"))
    return Embeddings(id_map, vecs)
