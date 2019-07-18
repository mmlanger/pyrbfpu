"""
Open questions:
- polynomial augmentation for rational and/or linear ansatz?
- polynomial augmentation useful/necessary?
- efficient svd algorithm for tridiagonal matrices?
- efficient eigenvalue/eigenvector calculation for tridiagonal matrices?
- efficient estimation of the condition number? (tridiagonal matrix version?)

"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import svd

import numba as nb

from pyrbfpu import util


@nb.njit
def rational_vector_eval(kernel, points, alpha, beta, x, param):
    dim = alpha.shape[1]
    num = np.zeros(dim)
    denom = np.zeros(dim)

    for i in range(alpha.shape[0]):
        phi = kernel(x, points[i], param)
        for k in range(dim):
            num[k] += alpha[i, k] * phi
            denom[k] += beta[i, k] * phi

    return num / denom


@nb.njit
def rational_scalar_eval(kernel, points, alpha, beta, x, param):
    num = 0.0
    denom = 0.0
    for i in range(alpha.shape[0]):
        phi = kernel(x, points[i], param)
        num += alpha[i] * phi
        denom += beta[i] * phi

    return num / denom


@nb.njit
def linear_vector_eval(kernel, points, coeffs, dummy, x, param):
    dim = coeffs.shape[1]
    result = np.zeros(dim)

    for i in range(coeffs.shape[0]):
        phi = kernel(x, points[i], param)
        for k in range(dim):
            result[k] += coeffs[i, k] * phi

    return result


@nb.njit
def linear_scalar_eval(kernel, points, coeffs, dummy, x, param):
    result = 0.0
    for i in range(coeffs.shape[0]):
        result += coeffs[i] * kernel(x, points[i], param)

    return result


@nb.njit
def linear_aug_scalar_eval(kernel, points, coeffs, dummy, x, param):
    n = points.shape[0]
    n_aug = coeffs.shape[0]
    result = 0.0

    for i in range(n):
        result += coeffs[i] * kernel(x, points[i], param)

    result += coeffs[n]

    for i in range(n + 1, n_aug):
        dim_idx = i - 1 - n
        result += coeffs[i] * x[dim_idx]

    return result


@nb.njit
def linear_aug_vector_eval(kernel, points, coeffs, dummy, x, param):
    dim = coeffs.shape[1]
    n = points.shape[0]
    n_aug = coeffs.shape[0]

    result = np.zeros(dim)

    for i in range(coeffs.shape[0]):
        phi = kernel(x, points[i], param)
        for k in range(dim):
            result[k] += coeffs[i, k] * phi

    for k in range(dim):
        result[k] += coeffs[n]

        for i in range(n + 1, n_aug):
            dim_idx = i - 1 - n
            result[k] += coeffs[i, k] * x[dim_idx]

    return result


class RBFInterpolation:
    def __init__(self, points, values, kernel, init_param, tol=1e-14):
        self.points = points
        self.values = values

        # self.optimize_values = np.linalg.norm(self.values, axis=1)
        if len(self.values.shape) > 1:
            f_norms = np.linalg.norm(self.values, axis=0)
            self.optimize_values = self.values[:, np.argmin(f_norms)]
        else:
            self.optimize_values = self.values

        self.param = init_param

        self.kernel = kernel
        self.tol = tol

        self.alpha = None
        self.beta = None

        self.eval_func = None

    def compute(self, rational=True):
        if rational:
            if len(self.values.shape) > 1:
                self.alpha = np.zeros(self.values.shape)
                self.beta = np.zeros(self.values.shape)

                for k in range(self.values.shape[1]):
                    alpha, beta = self.compute_rational(self.values[:, k])
                    self.alpha[:, k] = alpha
                    self.beta[:, k] = beta

                self.eval_func = rational_vector_eval
            else:
                self.alpha, self.beta = self.compute_rational(self.values)
                self.eval_func = rational_scalar_eval
        else:
            if len(self.values.shape) > 1:
                self.alpha = np.zeros(self.values.shape)

                for k in range(self.values.shape[1]):
                    self.alpha[:, k] = self.compute_linear(self.values[:, k])

                self.eval_func = linear_vector_eval
                # for k in range(self.values.shape[1]):
                #     self.alpha[:, k] = self.compute_augmented_linear(self.values[:, k])

                # self.eval_func = linear_aug_vector_eval

            else:
                self.alpha = self.compute_linear(self.values)
                self.eval_func = linear_scalar_eval
                # self.alpha = self.compute_augmented_linear(self.values)
                # self.eval_func = linear_aug_scalar_eval

    def compute_rational(self, f):
        if np.allclose(f, 0.0, rtol=0.0, atol=self.tol):
            return np.zeros(f.shape[0]), np.ones(f.shape[0])

        B = util.kernel_matrix(self.kernel, self.points, self.param)

        H, P = util.lanczos_decomposition(B, f, self.tol)
        U, s, Vh = svd(H, full_matrices=False)

        c = P @ (Vh[0] / s[0])

        alpha = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
        beta = c / s[0]

        return alpha, beta

    def compute_linear(self, f):
        B = util.kernel_matrix(self.kernel, self.points, self.param)
        # B -= 1e-3 * np.eye(f.shape[0])

        H, P = util.lanczos_decomposition(B, f, self.tol)
        Hinv = util.invert_symm_tridiag(H)
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return coeffs

    def compute_augmented_linear(self, f):
        B = util.augmented_kernel_matrix(self.kernel, self.points, self.param)
        n = f.shape[0]
        n_aug = B.shape[0]
        f_aug = np.hstack((f, np.zeros(n_aug - n)))

        coeffs = np.linalg.solve(B, f_aug)

        return coeffs

    def optimize_param(self):
        # print(
        #     "before {} with param={}".format(
        #         self.estimate_error(self.param), self.param
        #     )
        # )

        candidates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 5.0, 10.0]
        if self.param not in candidates:
            candidates.append(self.param)

        param_err = np.inf

        for param in candidates:
            try:
                new_param_err = self.estimate_error(param)
            except ZeroDivisionError:
                continue

            if new_param_err < param_err:
                self.param = param
                param_err = new_param_err

        # self.param = 0.5

        # res = minimize_scalar(
        #     self.estimate_error,
        #     bracket=(0.5 * best_param, best_param),
        #     method="brent",
        #     tol=1e-6,
        # )
        # self.param = res.x

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
        B -= 5e-13 * np.eye(f.shape[0])

        H, P = util.lanczos_decomposition(B, f, self.tol)
        Hinv = util.invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        scaled_coeffs = P @ Hinv[0, :]

        # print("reduction from {} to {}".format(f.shape[0], H.shape[0]))
        # if hasattr(self, "counter"):
        #     self.counter += 1
        # else:
        #     self.counter = 1

        return util.accumulate_error(scaled_coeffs, Binv)

    @property
    def computed(self):
        return self.alpha is not None

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )
