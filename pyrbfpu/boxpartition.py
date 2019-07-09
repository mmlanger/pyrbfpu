import math
from itertools import product
from multiprocessing import Pool

import numpy as np
import numba as nb


def boxindex(point, delta):
    return tuple(math.ceil(point[i] / delta) for i in range(len(point)))


@nb.njit
def boxcenter(index_tuple, delta):
    n = len(index_tuple)
    center = np.empty(n, dtype=np.float64)

    for i in range(n):
        center[i] = (index_tuple[i] - 0.5) * delta

    return center


def boxpartition(points, labels, delta):
    partition = {}

    for point, label in zip(points, labels):
        idx_tuple = tuple(boxindex(point, delta))
        indices = partition.get(idx_tuple, [])

        if indices:
            indices.append(label)
        else:
            partition[idx_tuple] = [label]

    return partition


def parallel_boxpartition(points, delta, n_proc=4):
    point_batches = np.array_split(points, n_proc)
    label_batches = np.array_split(np.arange(len(points)), n_proc)
    delta_batches = [delta for _ in range(n_proc)]

    batches = zip(point_batches, label_batches, delta_batches)
    # with Pool(n_proc) as p:
    #     partitions = p.starmap(boxpartition, batches)

    # *** serial version for debugger: ***
    from itertools import starmap
    partitions = list(starmap(boxpartition, batches))

    result_partition = partitions.pop()

    for partition in partitions:
        for key, idx_list in partition.items():
            try:
                result_partition[key].extend(idx_list)
            except KeyError:
                result_partition[key] = idx_list

    return result_partition


class BoxNeighbors:
    def __init__(self, dim):
        self.dim = dim
        self.shifts = []
        self.compute_level(1)

    def compute_level(self, max_level):
        known_shifts = len(self.shifts)
        for i in range(known_shifts, max_level + 1):
            new_shifts = set(product(range(-i, i + 1), repeat=self.dim))
            new_shifts = new_shifts.difference(*self.shifts)
            self.shifts.append(tuple(new_shifts))

    def neighbor_indices(self, center_tuple, level):
        if len(self.shifts) - 1 < level:
            self.compute_level(level)

        for shift_tuple in self.shifts[level]:
            yield tuple(i + s for i, s in zip(center_tuple, shift_tuple))
