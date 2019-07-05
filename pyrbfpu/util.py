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


def box_volume(box):
    vol = 1.0
    for k in range(box.shape[1]):
        vol *= box[0, k] - box[1, k]

    return vol


def box_shortest_side(box):
    length = abs(box[0, 0] - box[1, 0])
    for k in range(1, box.shape[1]):
        length = min(length, abs(box[0, k] - box[1, k]))

    return length


@nb.njit
def lanczos_decomposition(A, f, tol):
    n = A.shape[0]

    alpha = np.zeros(n, dtype=np.float64)
    beta = np.zeros(n, dtype=np.float64)

    q = f / np.linalg.norm(f)
    Q = np.zeros(A.shape)
    Q[:, 0] = q
    r = A @ q
    alpha[0] = np.dot(q, r)
    r -= alpha[0] * q
    beta[0] = np.linalg.norm(r)

    m = 1

    for j in range(1, n):
        v = q
        q = r / beta[j - 1]
        Q[:, j] = q
        r = A @ q - beta[j - 1] * v
        alpha[j] = np.dot(q, r)
        r -= alpha[j] * q
        Q_cur = Q[:, : j + 1].copy()
        r -= Q_cur @ (Q_cur.T @ r)
        beta[j] = np.linalg.norm(r)

        if beta[j] == 0.0 or abs(1.0 - np.sum(alpha[:m]) / n) < tol:
            break
        else:
            m = j + 1

    if n != m:
        Q = Q[:, :m]

    H = np.zeros((m, m), dtype=np.float64)
    H[0, 0] = alpha[0]
    for i in range(1, m):
        H[i, i] = alpha[i]
        H[i - 1, i] = beta[i - 1]
        H[i, i - 1] = beta[i - 1]

    return H, Q
