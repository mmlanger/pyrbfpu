from multiprocessing import Pool

import numpy as np
from scipy.special import gamma

import numba as nb

from pyrbfpu import kernels, util
from pyrbfpu import boxpartition as boxpart
from pyrbfpu.rationalrbf import RationalRBF
from pyrbfpu.vectorrbf import VectorRationalRBF


@nb.njit
def pyramid(point, center, box_length):
    diffs = np.abs(point - center) / box_length
    return max(1.0 - 2.0 * diffs.max(), 0.0)


def estimate_delta(points, min_cardinality):
    n_points = points.shape[0]
    point_dim = points.shape[1]

    bounding_box = util.bounding_box(points)
    bounding_vol = util.box_volume(bounding_box)

    density = n_points / bounding_vol
    domain_vol = min_cardinality / density

    num = gamma(point_dim / 2 + 1) * domain_vol
    denom = np.pi ** (point_dim / 2)
    delta = (num / denom) ** (1 / point_dim)

    return delta


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
        self.points = points
        self.values = values
        self.point_dim = points.shape[1]

        self.min_cardinality = min_cardinality
        if max_cardinality is None:
            self.max_cardinality = 2 * self.min_cardinality
        else:
            self.max_cardinality = max_cardinality

        if init_delta is None:
            self.delta = estimate_delta(self.points, self.min_cardinality)
        else:
            self.delta = init_delta
        self.box_length = 0.95 * 2 * np.sqrt(self.delta ** 2 / self.point_dim)

        if isinstance(rbf, str):
            if rbf == "imq":
                self.kernel_func = kernels.imq
            elif rbf == "gauss":
                self.kernel_func = kernels.gauss
            elif rbf == "wendland":
                self.kernel_func = kernels.wendland
            elif rbf == "wendland_C4":
                self.kernel_func = kernels.wendland_C4
            else:
                raise ValueError("Unknown rbf function!")
        else:
            self.kernel_func = rbf

        # @nb.njit
        # def scale_func(x, u):
        #     return u * np.linalg.norm(x)

        # self.kernel = kernels.generate_vskernel(self.kernel_func, scale_func)
        self.kernel = kernels.generate_kernel(self.kernel_func)

        if len(self.values.shape) > 1:
            self.interpolant_type = VectorRationalRBF
        else:
            self.interpolant_type = RationalRBF

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
            interpolator = self.interpolant_type(
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
