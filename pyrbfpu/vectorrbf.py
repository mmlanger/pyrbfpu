import numpy as np
from scipy.optimize import minimize_scalar

import numba as nb

from pyrbfpu import util


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

        # self.optimize_values = np.linalg.norm(self.values, axis=1)
        f_norms = np.linalg.norm(self.values, axis=0)
        self.optimize_values = self.values[:, np.argmin(f_norms)]

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

            H, P = util.lanczos_decomposition(B, f, self.tol)
            U, s, Vh = np.linalg.svd(H, full_matrices=False)

            c = P @ (Vh[0] / s[0])

            self.alpha[:, k] = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
            self.beta[:, k] = c / s[0]

    def optimize_param(self):
        print(
            "before {} with param={}".format(
                self.estimate_error(self.param), self.param
            )
        )
        candidates = [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 5.0]
        if self.param not in candidates:
            candidates.append(self.param)
        errors = [(eps, self.estimate_error(eps)) for eps in candidates]
        self.param, err = min(errors, key=lambda x: x[1])
        # res = minimize_scalar(
        #     self.estimate_error,
        #     bracket=(1e-6, 5.0),
        #     bounds=(1e-6, 5.0),
        #     method="Bounded",
        #     tol=1e-3,
        # )
        # self.param = res.x
        print(
            "after  {} with param={}".format(
                self.estimate_error(self.param), self.param
            )
        )

    def estimate_error(self, param):
        f = self.optimize_values
        B = util.kernel_matrix(self.kernel, self.points, param)

        H, P = util.lanczos_decomposition(B, f, self.tol)
        Hinv = util.invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        scaled_coeffs = P @ Hinv[0, :]

        return util.accumulate_error(scaled_coeffs, Binv)

    @property
    def computed(self):
        return self.alpha is not None

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )
