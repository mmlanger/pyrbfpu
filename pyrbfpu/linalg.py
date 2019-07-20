import numpy as np
import numba as nb


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
