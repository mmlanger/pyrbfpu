import numpy as np
from scipy.optimize import minimize

import numba as nb

from pyrbfpu import util


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

        if np.allclose(self.values, 0.0, rtol=0.0, atol=self.tol):
            self.eval_func = lambda x: 0.0

    def compute(self):
        f = self.values

        B = util.kernel_matrix(self.kernel, self.points, self.param)
        H, P = util.lanczos_decomposition(B, f, self.tol)
        # print(H.shape[0])
        U, s, Vh = np.linalg.svd(H, full_matrices=False)

        c = P @ (Vh[0] / s[0])

        self.alpha = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
        self.beta = c / s[0]

        self.eval_func = interp_eval

    def compute_nonrational(self):
        f = self.values
        B = util.kernel_matrix(self.kernel, self.points, self.param)

        H, P = util.lanczos_decomposition(B, f, self.tol)
        Hinv = util.invert_symm_tridiag(H)
        self.alpha = (P @ Hinv[0, :]) * np.linalg.norm(f)

        @nb.njit
        def interp_eval(kernel, points, coeffs, dummy, x, param):
            result = 0.0
            for i in range(points.shape[0]):
                result += coeffs[i] * kernel(x, points[i], param)

            return result

        self.eval_func = interp_eval

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

    def estimate_error(self, param_arr):
        f = self.values
        B = util.kernel_matrix(self.kernel, self.points, param_arr[0])

        H, P = util.lanczos_decomposition(B, f, self.tol)
        Hinv = util.invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        scaled_coeffs = P @ Hinv[0, :]

        return util.accumulate_error(scaled_coeffs, Binv)

    @property
    def computed(self):
        return self.eval_func is not None

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )
