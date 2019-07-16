from multiprocessing import Pool

import numpy as np
from scipy.special import gamma

import numba as nb

from pyrbfpu import kernels, util
from pyrbfpu import boxpartition as boxpart
from pyrbfpu.rbf import RBFInterpolation


@nb.njit
def pyramid(point, center, box_length):
    diffs = np.abs(point - center) / box_length
    return max(1.0 - 2.0 * diffs.max(), 0.0)


class RatRBFPartUnityInterpolation:
    def __init__(
        self,
        points,
        values,
        min_cardinality=100,
        max_cardinality=None,
        weight_overlap=0.01,
        init_delta=None,
        rbf="imq",
        tol=1e-14,
    ):
        if len(points.shape) == 1:
            self.points = points.reshape((values.shape[0], 1))
        else:
            self.points = points

        self.point_dim = self.points.shape[1]
        self.values = values

        self.min_cardinality = min(min_cardinality, self.points.shape[0])
        if max_cardinality is None:
            self.max_cardinality = 2 * self.min_cardinality
        else:
            self.max_cardinality = max_cardinality

        if init_delta is None:
            bounding_box = util.bounding_box(self.points)
            bounding_vol = util.box_volume(bounding_box)

            density = self.points.shape[0] / bounding_vol
            domain_vol = self.min_cardinality / density

            num = gamma(self.point_dim / 2 + 1) * domain_vol
            denom = np.pi ** (self.point_dim / 2)
            self.delta = (num / denom) ** (1 / self.point_dim)
        else:
            self.delta = init_delta

        self.box_length = 0.95 * 2 * np.sqrt(self.delta ** 2 / self.point_dim)

        if isinstance(rbf, str):
            if hasattr(kernels, rbf):
                self.kernel_func = getattr(kernels, rbf)
            else:
                raise ValueError("Unknown rbf function!")
        else:
            self.kernel_func = rbf

        self.kernel = kernels.generate_kernel(self.kernel_func)

        self.tol = tol
        self.weight_overlap = weight_overlap

        self.neighbors = boxpart.BoxNeighbors(self.point_dim)

        self.boxpartition = boxpart.parallel_boxpartition(self.points, self.box_length)
        self.subdomains = {}
        self.box_overlaps = {}

    def domain_decomposition(self):
        for idx_key in self.boxpartition:
            self.subdomain_setup(idx_key)

    def subdomain_setup(self, idx_key):
        contained_indices = self.boxpartition.get(idx_key, [])

        local_center = boxpart.boxcenter(idx_key, self.box_length)
        surr_indices = []
        surr_dists = []

        # collect surrounding indices of current box to fulfill minimal cardinality
        level = 0
        inside_box = len(contained_indices)
        inside_sphere = 0

        while inside_sphere < self.min_cardinality:
            level += 1

            for neighbor_key in self.neighbors.neighbor_indices(idx_key, level):
                surr_indices.extend(self.boxpartition.get(neighbor_key, []))

            if not surr_indices:
                continue

            surr_points = np.array([self.points[i] for i in surr_indices])
            surr_dists = util.center_distance_arr(local_center, surr_points)

            max_delta = (0.5 + level) * self.box_length
            inside_surrounding = util.count_inside_sphere(surr_dists, max_delta)
            inside_sphere = inside_box + inside_surrounding

        surr_sorted = np.argsort(surr_dists)
        surr_dists = surr_dists[surr_sorted]
        surr_indices = np.array(surr_indices)[surr_sorted]

        cardinality = inside_box
        local_indices = contained_indices.copy()
        for point_idx, local_delta in zip(surr_indices, surr_dists):
            if local_delta < self.delta or cardinality < self.min_cardinality:
                local_indices.append(point_idx)
                cardinality += 1
            else:
                break

        local_points = np.array([self.points[i] for i in local_indices])
        local_values = np.array([self.values[i] for i in local_indices])

        param = 0.4 * np.sqrt(len(local_indices)) / local_delta

        if cardinality <= self.max_cardinality:
            interpolator = RBFInterpolation(
                local_points, local_values, self.kernel, param, self.tol
            )
        else:
            interpolator = RatRBFPartUnityInterpolation(
                local_points,
                local_values,
                min_cardinality=self.min_cardinality,
                max_cardinality=self.max_cardinality,
                weight_overlap=self.weight_overlap,
                rbf=self.kernel_func,
                tol=self.tol,
            )

        self.subdomains[idx_key] = interpolator
        return interpolator

    @property
    def computed(self):
        # for lazy evaluation only the box partition is necessary
        return True

    def __call__(self, point):
        point = np.asarray(point)
        box_idx = tuple(boxpart.boxindex(point, self.box_length))

        try:
            overlaps = self.box_overlaps[box_idx]
        except KeyError:
            overlaps = []
            for level in range(2):
                for idx_key in self.neighbors.neighbor_indices(box_idx, level):
                    if level > 0 and idx_key not in self.boxpartition:
                        continue
                    center = boxpart.boxcenter(idx_key, self.box_length)
                    try:
                        interpolator = self.subdomains[idx_key]
                    except KeyError:
                        interpolator = self.subdomain_setup(idx_key)
                    overlaps.append((center, interpolator))

            self.box_overlaps[box_idx] = overlaps

        effective_box_length = self.box_length * (1.0 + self.weight_overlap)
        weight_sum = 0.0
        f = 0.0

        for (center, interpolator) in overlaps:
            weight = pyramid(point, center, effective_box_length)
            # print(weight)
            weight = weight ** 2
            if weight > 0.0:
                weight_sum += weight

                if not interpolator.computed:
                    interpolator.optimize_param()
                    interpolator.compute()

                f += weight * interpolator(point)

        if weight_sum == 0.0:
            raise ValueError("Point not inside partition!")

        return f / weight_sum
