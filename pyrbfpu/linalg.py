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


def TDMAsolver(a, b, c, d):
    nf = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def largest_eigenpair_tridiag(A):
    N = A.shape[0] - 1

    m = max(A[0, 0], A[0, 1])
    for n in range(1, N):
        m = max(m, A[n, n - 1], A[n, n], A[n, n + 1])
    m = max(m, A[N, N - 1], A[N, N])

    Q = np.zeros((N + 1, N + 1))

    Q[0, 0] = A[n, n] - m
    Q[0, 1] = A[n, n - 1]
    for n in range(1, N):
        Q[n, n] = A[n, n] - m
        Q[n, n - 1] = A[n, n - 1]
        Q[n, n + 1] = A[n, n + 1]
    Q[N, N] = A[N, N] - m
    Q[N, N - 1] = A[N, N - 1]

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)

    b[0] = Q[0, 1]
    c[0] = -Q[0, 0] - b[0]
    m = max(Q[0, 0], b[0])

    for n in range(1, N):
        a[n] = Q[n, n - 1]
        b[n] = Q[n, n + 1]
        c[n] = -(Q[n, n] + a[n] + b[n])

        m = max(m, a[n], Q[n, n], b[n])

    a[N] = Q[N, N - 1]
    b[N] = 1.0
    c[N] = -Q[N, N] - a[N]

    m = max(m, a[N], Q[N, N])

    r = np.zeros(N + 1)
    h = np.zeros(N + 2)

    r[0] = 1.0 + a[0] / b[0]
    h[0] = 1.0
    for n in range(1, N):
        r[n] = 1.0 + (a[n] + c[n]) / b[n] - a[n] / (b[n] * r[n - 1])
        h[n] = h[n - 1] * r[n - 1]
    h[N] = h[N - 1] * r[N - 1]
    h[N + 1] = c[N] * h[N] + a[N] * (h[N] - h[N - 1])

    a_tilde = np.zeros(N + 1)
    b_tilde = np.zeros(N + 1)
    for n in range(N):
        # Q_tilde[i,i+1] = Q[i,i+1] * r[i]
        # Q_tilde[i+1,i] = Q[i+1,i] / r[i]
        a_tilde[n + 1] = Q[n + 1, n] / r[n]
        b_tilde[n] = Q[n, n + 1] * r[n]

    nu = np.zeros(N + 1)
    nu[0] = 1.0
    for n in range(1, N + 1):
        nu[n] = nu[n - 1] * b_tilde[n - 1] / a_tilde[n]

    phi = np.zeros(N + 1)
    phi[N] = 1.0 / (nu[N] * b_tilde[N])
    for n in range(N - 1, -1, -1):
        phi[n] = phi[n + 1] + 1.0 / (nu[n] * b_tilde[n])

    # v /= np.sqrt(np.dot(v, v))

    W = np.zeros(N + 1)
    T = np.zeros(N)

    for n in range(N):
        W[n] = nu[n] * phi[n]
        T[n] = phi[n + 1] / phi[n]
    W[N] = 1.0 / b_tilde[N]

    eta1 = np.zeros(N + 1)
    eta1[0] = W[0]
    for n in range(1, N + 1):
        eta1[n] = W[n] + np.sqrt(T[n - 1]) * eta1[n - 1]

    eta2 = np.zeros(N + 1)
    eta2[N] = 0.0
    for n in range(N - 1, -1, -1):
        eta2[n] = T[n] * (eta2[n + 1] + W[n + 1])

    delta = eta1[0] + eta2[0]
    for k in range(1, N + 1):
        delta = max(delta, eta1[k] + eta2[k])

    # construct initial guess for eigenpair
    z = 1.0 / delta
    v = np.sqrt(phi)
    v /= np.dot(v, v)

    # iterate eigenpair until convergence
    xi1 = np.zeros(N + 1)
    xi2 = np.zeros(N + 1)
    xi1[0] = W[0]
    xi2[N] = 0.0

    mat_superdiag = -b_tilde
    mat_subdiag = -a_tilde

    diag_base = a_tilde + b_tilde

    for n in range(10000):
        z_prev = z
        mat_diag = diag_base - z

        # iterate eigenvector
        v = TDMAsolver(mat_subdiag, mat_diag, mat_superdiag, v)
        v /= np.dot(v, v)

        # estimate new delta
        for k in range(1, N + 1):
            xi1[k] = W[k] + (v[k - 1] / v[k]) * T[k - 1] * xi1[k - 1]

        for k in range(N - 1, -1, -1):
            xi2[k] = (v[k + 1] / v[k]) * (xi2[k + 1] + W[k + 1])

        delta = xi1[0] + xi2[0]
        for k in range(1, N + 1):
            delta = max(delta, xi1[k] + xi2[k])

        z = 1.0 / delta

        # check if converged
        if abs(z - z_prev) < 1e-8:
            break

    eigval = m - z
    eigvec = h * v
    return eigval, eigvec
