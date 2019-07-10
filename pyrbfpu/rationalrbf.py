import numpy as np
from scipy.linalg import lu, ldl, solve_triangular
from scipy.sparse.linalg import eigsh, spsolve_triangular
from scipy.optimize import minimize

import numba as nb

from pyrbfpu.util import lanczos_decomposition, invert_symm_tridiag


@nb.njit
def kernel_matrix(kernel, points, param):
    n = points.shape[0]

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = kernel(points[i], points[i], param)

        for j in range(i + 1, n):
            val = kernel(points[i], points[j], param)
            if val != 0.0:
                A[i, j] = val
                A[j, i] = val

    return A


@nb.njit
def interp_eval(kernel, points, alpha, beta, x, param):
    num = 0.0
    denom = 0.0
    for i in range(points.shape[0]):
        phi = kernel(x, points[i], param)
        num += alpha[i] * phi
        denom += beta[i] * phi

    return num / denom


class RationalRBF:
    def __init__(self, points, values, kernel, init_param, tol=1e-14):
        self.points = points
        self.values = values

        self.param = init_param

        self.kernel = kernel
        self.tol = tol

        self.alpha = None
        self.beta = None

        self.eval_func = None

        if np.allclose(self.values, 0.0, rtol=0.0, atol=1e-14):
            self.eval_func = lambda x: 0.0

    def compute(self):
        f = self.values

        B = kernel_matrix(self.kernel, self.points, self.param)
        H, P = lanczos_decomposition(B, f, self.tol)
        #print(H.shape[0])
        U, s, Vh = np.linalg.svd(H, full_matrices=False)

        c = P @ (Vh[0] / s[0])

        self.alpha = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
        self.beta = c / s[0]

        self.eval_func = interp_eval

    def optimize_param(self):
        #print("before {}".format(self.estimate_error([self.param])))
        res = minimize(
            self.estimate_error,
            self.param,
            method="Nelder-Mead",
            options=dict(maxiter=30),
        )
        self.param = res.x[0]
        #print("after  {}".format(self.estimate_error([self.param])))

    def estimate_error(self, param):
        f = self.values
        B = kernel_matrix(self.kernel, self.points, param[0])

        H, P = lanczos_decomposition(B, f, self.tol)
        Hinv = invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        coeffs = Binv @ f

        return np.linalg.norm(coeffs / np.diagonal(Binv))

    @property
    def computed(self):
        return self.eval_func is not None

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )


class LegacyRationalRBF:
    def __init__(self, points, values, kernel, tol=1e-14):
        self.points = points
        self.values = values

        self.kernel = kernel
        self.tol = tol

        self.alpha = None
        self.beta = None

        self.eval_func = None

        if np.allclose(self.values, 0.0, rtol=0.0, atol=1e-14):
            self.eval_func = lambda x: 0.0

    def compute(self):
        f = self.values

        B = kernel_matrix(self.kernel, self.points)
        H, P = lanczos_decomposition(B, f, self.tol)
        # print(H.shape[0])
        U, s, Vh = np.linalg.svd(H, full_matrices=False)

        c = P @ (Vh[0] / s[0])

        self.alpha = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
        self.beta = c / s[0]

        self.eval_func = interp_eval

    def compute2(self):
        f = self.values

        B = kernel_matrix(self.kernel, self.points, 5e-15)

        lu, d, perm = ldl(B)
        L = lu[perm, :]
        D = d[perm, :]
        DLT = D @ L.T

        _, eig_vecs = eigsh(B, k=1, maxiter=20000, which="LM", v0=f)

        q = eig_vecs[:, 0]
        p = f * q

        solve = solve_triangular
        self.alpha = solve(DLT, solve(L, p, lower=True), lower=False)
        self.beta = solve(DLT, solve(L, q, lower=True), lower=False)

        @nb.njit
        def interp_eval(kernel, points, alpha, beta, x):
            num = 0.0
            denom = 0.0
            for i in range(points.shape[0]):
                phi = kernel(x, points[i])
                num += alpha[i] * phi
                denom += beta[i] * phi

            return num / denom

        self.eval_func = interp_eval

    def compute3(self):
        f = self.values
        B = kernel_matrix(self.kernel, self.points)

        H, P = lanczos_decomposition(B, f, self.tol)
        U, s, Vh = np.linalg.svd(H, full_matrices=False)

        self.alpha = P @ Vh.T @ (U[0] * (np.linalg.norm(f) / s))

        @nb.njit
        def interp_eval(kernel, points, coeffs, dummy, x):
            result = 0.0
            for i in range(points.shape[0]):
                result += coeffs[i] * kernel(x, points[i])

            return result

        self.eval_func = interp_eval

    @property
    def computed(self):
        return self.eval_func is not None

    def __call__(self, x):
        return self.eval_func(self.kernel, self.points, self.alpha, self.beta, x)
