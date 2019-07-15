import numpy as np
from scipy.optimize import minimize_scalar

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
        U, s, Vh = np.linalg.svd(H, full_matrices=False)

        c = P @ (Vh[0] / s[0])

        self.alpha = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
        self.beta = c / s[0]

        self.eval_func = interp_eval

    def compute_nonrational(self):
        f = self.values
        B = util.kernel_matrix(self.kernel, self.points, self.param)
        #B -= 1e-4 * np.eye(f.shape[0])

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
        f = self.values
        B = util.kernel_matrix(self.kernel, self.points, param)

        H, P = util.lanczos_decomposition(B, f, self.tol)
        Hinv = util.invert_symm_tridiag(H)
        K = P @ Hinv
        Binv = K @ P.T
        scaled_coeffs = K[:, 0]

        # print("reduction from {} to {}".format(f.shape[0], H.shape[0]))
        # if hasattr(self, "counter"):
        #     self.counter += 1
        # else:
        #     self.counter = 1
      
        return util.accumulate_error(scaled_coeffs, Binv)

    @property
    def computed(self):
        return self.eval_func is not None

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )
