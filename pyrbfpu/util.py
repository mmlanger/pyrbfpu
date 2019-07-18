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
        vol *= abs(box[0, k] - box[1, k])

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


@nb.njit
def invert_symm_tridiag(T):
    n = T.shape[0]
    s = np.zeros((n, n))

    a = np.zeros(n)
    b = np.zeros(n - 1)

    a[0] = T[0, 0]
    for i in range(1, n):
        a[i] = T[i, i]
        b[i - 1] = -T[i, i - 1]

    v = np.zeros(n)
    u = np.zeros(n)

    v[0] = 1.0
    v[1] = a[0] / b[0]
    for i in range(2, n):
        v[i] = (a[i - 1] * v[i - 1] - b[i - 2] * v[i - 2]) / b[i - 1]

    u[n - 1] = 1.0 / (-b[n - 2] * v[n - 2] + a[n - 1] * v[n - 1])
    for i in range(n - 2, 0, -1):
        u[i] = (1.0 + b[i] * v[i] * u[i + 1]) / (a[i] * v[i] - b[i - 1] * v[i - 1])
    u[0] = (1.0 + b[0] * v[0] * u[1]) / (a[0] * v[0])

    for i in range(n):
        for j in range(i, n):
            val = u[j] * v[i]
            s[i, j] = val
            s[j, i] = val

    return s


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
        for j in range(0, n):
            val = points[j, dim_idx]
            A[i, j] = val
            A[j, i] = val

    return A


@nb.njit
def accumulate_error(coeffs, Binv):
    result = 0.0
    for i in range(coeffs.shape[0]):
        error = coeffs[i] / Binv[i, i]
        result += error * error

    return result
