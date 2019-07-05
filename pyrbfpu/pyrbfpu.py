#%%

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
        min_cardinality=50,
        weight_overlap=0.01,
        init_delta=None,
        rbf="imq",
        tol=1e-14,
    ):
        self.points = points
        self.values = values
        self.point_dim = points.shape[1]

        self.min_cardinality = min_cardinality
        if init_delta is None:
            self.delta = estimate_delta(self.points, self.min_cardinality)
        else:
            self.delta = init_delta
        self.box_length = 0.9 * 2 * np.sqrt(self.delta ** 2 / self.point_dim)

        if rbf == "imq":
            self.kernel = kernels.imq
        elif rbf == "gauss":
            self.kernel = kernels.gauss
        elif rbf == "wendland":
            self.kernel = kernels.wendland
        elif rbf == "wendland_C4":
            self.kernel = kernels.wendland_C4
        else:
            raise ValueError("Unknown rbf function!")

        self.scale_func = kernels.generate_scale_func_v2(2.0)
        self.vskernel = kernels.generate_vskernel(self.kernel, self.scale_func)

        if len(self.values.shape) > 1:
            self.interpolant_type = VectorRationalRBF
        else:
            self.interpolant_type = RationalRBF

        self.tol = tol
        self.weight_overlap = weight_overlap

        self.boxpartition = None
        self.subdomains = None
        self.overlapping_subdomains = None

    def domain_decomposition(self):
        self.boxpartition = boxpart.parallel_boxpartition(self.points, self.box_length)

        self.subdomains = {}
        self.overlapping_subdomains = {key: [] for key in self.boxpartition}
        overlapping_keys = {key: {key} for key in self.boxpartition}

        neighbors = boxpart.BoxNeighbors(self.point_dim)

        for idx_key, contained_indices in self.boxpartition.items():
            if not contained_indices:
                continue

            local_center = boxpart.boxcenter(idx_key, self.box_length)
            surr_indices = []
            surr_dists = []

            # collect surrounding indices of current box to fulfill minimal cardinality
            level = 0
            inside_box = len(contained_indices)
            inside_sphere = 0

            for neighbor_key in neighbors.neighbor_indices(idx_key, 1):
                try:
                    overlapping_keys[neighbor_key].add(idx_key)
                except KeyError:
                    pass

            while inside_sphere < self.min_cardinality:
                level += 1
                for neighbor_key in neighbors.neighbor_indices(idx_key, level):
                    surr_indices.extend(self.boxpartition.get(neighbor_key, []))

                surr_points = np.array([self.points[i] for i in surr_indices])
                surr_dists = util.center_distance_arr(local_center, surr_points)

                max_delta = (0.5 + level) * self.box_length
                inside_surrounding = util.count_inside_sphere(surr_dists, max_delta)
                inside_sphere = inside_box + inside_surrounding

            surr_sorted = np.argsort(surr_dists)

            cardinality = inside_box
            local_indices = contained_indices.copy()
            for i in range(surr_sorted.shape[0]):
                local_delta = surr_dists[i]
                if local_delta < self.delta or cardinality < self.min_cardinality:
                    local_indices.append(local_indices[i])
                    cardinality += 1
                else:
                    break

            local_points = np.array([self.points[i] for i in local_indices])
            local_values = np.array([self.values[i] for i in local_indices])

            interpolator = self.interpolant_type(
                local_points, local_values, self.vskernel, self.tol
            )

            self.subdomains[idx_key] = (local_center, interpolator)

        for box_key in self.boxpartition:
            domains = [self.subdomains[key] for key in overlapping_keys[box_key]]
            self.overlapping_subdomains[box_key] = domains

    def __call__(self, point):
        point = np.asarray(point)
        box_idx = tuple(boxpart.boxindex(point, self.box_length))

        effective_box_length = self.box_length * (1.0 + self.weight_overlap)
        weight_sum = 0.0
        f = 0.0

        for (center, interpolator) in self.overlapping_subdomains[box_idx]:
            weight = pyramid(point, center, effective_box_length)
            # print(weight)
            weight = weight ** 2
            if weight > 0.0:
                weight_sum += weight

                if not interpolator.computed:
                    interpolator.compute()

                f += weight * interpolator(point)

        if weight_sum == 0.0:
            return ValueError("Point not inside partition!")

        return f / weight_sum


#%%
