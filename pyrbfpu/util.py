import numpy as np
import numba as nb


@nb.njit
def dist(x1, x2):
    n = x1.shape[0]

    r_sqr = 0.0
    for i in range(n):
        diff = x1[i] - x2[i]
        r_sqr += diff * diff

    return np.sqrt(r_sqr)


@nb.njit
def center_distance_arr(center, points):
    n = points.shape[0]
    dists = np.empty(n, dtype=np.float64)

    for i in range(n):
        dists[i] = dist(center, points[i])

    return dists


@nb.njit
def count_inside_sphere(dists_to_center, delta):
    count = 0
    for d in dists_to_center:
        if d < delta:
            count += 1

    return count


@nb.njit
def bounding_box(points):
    point_dim = points.shape[1]
    box = np.zeros((2, point_dim))

    for k in range(point_dim):
        val = points[0, k]
        box[0, k] = val
        box[1, k] = val

    for i in range(1, points.shape[0]):
        for k in range(point_dim):
            val = points[i, k]
            box[0, k] = min(val, box[0, k])
            box[1, k] = max(val, box[1, k])

    return box


@nb.njit
def apply_scaling(point, shifts, scales):
    point_dim = point.shape[0]
    rescaled_point = np.zeros(point_dim)

    for k in range(point_dim):
        rescaled_point[k] = (point[k] + shifts[k]) / scales[k]

    return rescaled_point


@nb.njit
def reverse_scaling(point, shifts, scales):
    point_dim = point.shape[0]
    rescaled_point = np.zeros(point_dim)

    for k in range(point_dim):
        rescaled_point[k] = point[k] * scales[k] - shifts[k]

    return rescaled_point


def box_volume(box):
    vol = 1.0
    for k in range(box.shape[1]):
        vol *= abs(box[0, k] - box[1, k])

    return vol


def box_shortest_side(box):
    length = abs(box[0, 0] - box[1, 0])
    for k in range(1, box.shape[1]):
        length = min(length, abs(box[0, k] - box[1, k]))

    return length


@nb.njit
def kernel_matrix(kernel, points, param):
    n = points.shape[0]

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        point = points[i]
        A[i, i] = kernel(point, point, param)

        for j in range(i + 1, n):
            val = kernel(point, points[j], param)
            if val != 0.0:
                A[i, j] = val
                A[j, i] = val

    return A


@nb.njit
def augmented_kernel_matrix(kernel, points, param):
    dim = points.shape[1]
    n = points.shape[0]
    n_aug = n + dim + 1

    A = np.zeros((n_aug, n_aug), dtype=np.float64)
    for i in range(n):
        point = points[i]
        A[i, i] = kernel(point, point, param)

        for j in range(i + 1, n):
            val = kernel(point, points[j], param)
            if val != 0.0:
                A[i, j] = val
                A[j, i] = val

    for j in range(0, n):
        A[n, j] = 1.0
        A[j, n] = 1.0

    for i in range(n + 1, n_aug):
        dim_idx = i - 1 - n
        for j in range(n):
            val = points[j, dim_idx]
            A[i, j] = val
            A[j, i] = val

    return A


@nb.njit
def loocv_error(coeffs, Binv):
    result = 0.0
    for i in range(coeffs.shape[0]):
        error = coeffs[i] / Binv[i, i]
        result = max(result, abs(error))

    return result


@nb.njit
def loocv_error2(coeffs, Binv):
    result = 0.0
    for i in range(coeffs.shape[0]):
        error = coeffs[i] / Binv[i, i]
        result += error * error

    return result
