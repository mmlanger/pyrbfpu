import numpy as np
from scipy.optimize import minimize

import numba as nb

from pyrbfpu.util import lanczos_decomposition, invert_symm_tridiag


@nb.njit
def interp_eval(kernel, points, alpha, beta, x, param):
    dim = alpha.shape[1]
    num = np.zeros(dim)
    denom = np.zeros(dim)

    for i in range(points.shape[0]):
        phi = kernel(x, points[i], param)
        for k in range(dim):
            num[k] += alpha[i, k] * phi
            denom[k] += beta[i, k] * phi

    return num / denom


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


class VectorRationalRBF:
    def __init__(self, points, values, kernel, init_param, tol=1e-14):
        self.points = points
        self.values = values

        self.optimize_values = np.linalg.norm(self.values, axis=1)

        self.param = init_param

        self.kernel = kernel
        self.tol = tol

        self.alpha = None
        self.beta = None

        self.eval_func = interp_eval

    def compute(self):
        f = self.values
        B = kernel_matrix(self.kernel, self.points, self.param)

        self.alpha = np.zeros(f.shape)
        self.beta = np.zeros(f.shape)

        for k in range(self.values.shape[1]):
            f = self.values[:, k]

            if np.allclose(f, 0.0, rtol=0.0, atol=self.tol):
                self.alpha[:, k] = 0.0
                self.beta[:, k] = 1.0
                continue

            H, P = lanczos_decomposition(B, f, self.tol)
            U, s, Vh = np.linalg.svd(H, full_matrices=False)

            c = P @ (Vh[0] / s[0])

            self.alpha[:, k] = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
            self.beta[:, k] = c / s[0]

    def optimize_param(self):
        # print("before {}".format(self.estimate_error([self.param])))
        res = minimize(
            self.estimate_error,
            self.param,
            method="L-BFGS-B",
            bounds=[(1e-6, 200.0)],
            options=dict(maxiter=150),
        )
        self.param = res.x[0]
        # print("after  {}".format(self.estimate_error([self.param])))

    def estimate_error(self, param):
        f = self.optimize_values
        B = kernel_matrix(self.kernel, self.points, param[0])

        H, P = lanczos_decomposition(B, f, self.tol)
        Hinv = invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return np.linalg.norm(coeffs / np.diagonal(Binv))

    @property
    def computed(self):
        return self.alpha is not None

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )
