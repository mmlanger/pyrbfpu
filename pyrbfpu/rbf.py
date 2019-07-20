"""
Open questions:
- polynomial augmentation for rational and/or linear ansatz? (rational has issues with constant functions!)
- polynomial augmentation useful/necessary?
- efficient svd algorithm for tridiagonal matrices?
- efficient eigenvalue/eigenvector calculation for tridiagonal matrices?
- efficient estimation of the condition number? (tridiagonal matrix version?)
- noise free error estimate? different error estimate? (needs to be fast)

"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import svd

import numba as nb

from pyrbfpu import util
from pyrbfpu.linalg import lanczos_decomposition, invert_symm_tridiag


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
def linear_vector_eval(kernel, points, coeffs, x, param):
    dim = coeffs.shape[1]
    result = np.zeros(dim)

    for i in range(coeffs.shape[0]):
        phi = kernel(x, points[i], param)
        for k in range(dim):
            result[k] += coeffs[i, k] * phi

    return result


@nb.njit
def linear_scalar_eval(kernel, points, coeffs, x, param):
    result = 0.0
    for i in range(coeffs.shape[0]):
        result += coeffs[i] * kernel(x, points[i], param)

    return result


@nb.njit
def linear_aug_scalar_eval(kernel, points, coeffs, x, param):
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
def linear_aug_vector_eval(kernel, points, coeffs, x, param):
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


class RBFInterpolationBase:
    def __init__(self, points, values, kernel, init_param):
        self.points = points
        self.values = values

        if len(self.values.shape) > 1:
            f_norms = np.linalg.norm(self.values, axis=0)
            self.optimize_values = self.values[:, np.argmin(f_norms)]
        else:
            self.optimize_values = self.values

        self.param = init_param

        self.kernel = kernel
        self.eval_func = None

    def compute(self):
        raise NotImplementedError

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

        # self.param = 0.01

        res = minimize_scalar(
            self.estimate_error,
            bracket=(0.5 * self.param, self.param),
            method="brent",
            tol=1e-6,
        )
        self.param = res.x

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
        raise NotImplementedError

    @property
    def computed(self):
        return self.eval_func is not None

    def __call__(self, x):
        raise NotImplementedError


class RBFInterpolationRational(RBFInterpolationBase):
    def __init__(self, points, values, kernel, init_param, tol=1e-14, smooth=None):
        super().__init__(points, values, kernel, init_param)
        self.tol = tol
        self.smooth = None

        self.alpha = None
        self.beta = None

    def compute(self):
        if len(self.values.shape) > 1:
            self.alpha = np.zeros(self.values.shape)
            self.beta = np.zeros(self.values.shape)

            for k in range(self.values.shape[1]):
                alpha, beta = self.compute_coeffs(self.values[:, k])
                self.alpha[:, k] = alpha
                self.beta[:, k] = beta

            self.eval_func = rational_vector_eval
        else:
            self.alpha, self.beta = self.compute_coeffs(self.values)
            self.eval_func = rational_scalar_eval

    def compute_coeffs(self, f):
        if f.max() - f.min() < self.tol:
            return np.full(f.shape[0], np.mean(f)), np.ones(f.shape[0])

        B = util.kernel_matrix(self.kernel, self.points, self.param)

        H, P = lanczos_decomposition(B, f, self.tol)
        U, s, Vh = svd(H, full_matrices=False)

        if self.smooth is not None:
            s = (s ** 2 + self.smooth ** 2) / s

        c = P @ (Vh[0] / s[0])

        alpha = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
        beta = c / s[0]

        return alpha, beta

    def estimate_error(self, param):
        # TODO: rational version
        f = self.optimize_values
        B = util.kernel_matrix(self.kernel, self.points, param)

        H, P = lanczos_decomposition(B, f, self.tol)
        Hinv = invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return util.loocv_error(coeffs, Binv)

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )



class RBFInterpolationRescaledLocalized(RBFInterpolationBase):
    def __init__(self, points, values, kernel, init_param, tol=1e-14):
        super().__init__(points, values, kernel, init_param)
        self.tol = tol

        self.alpha = None
        self.beta = None

    def compute(self):
        if len(self.values.shape) > 1:
            self.alpha = np.zeros(self.values.shape)
            self.beta = np.zeros(self.values.shape)

            for k in range(self.values.shape[1]):
                alpha, beta = self.compute_coeffs(self.values[:, k])
                self.alpha[:, k] = alpha
                self.beta[:, k] = beta

            self.eval_func = rational_vector_eval
        else:
            self.alpha, self.beta = self.compute_coeffs(self.values)
            self.eval_func = rational_scalar_eval

    def compute_coeffs(self, f):
        B = util.kernel_matrix(self.kernel, self.points, self.param)

        alpha = np.linalg.solve(B, f)
        beta = np.linalg.solve(B, np.ones(f.shape))

        return alpha, beta

    def estimate_error(self, param):
        # TODO: rational version
        f = self.optimize_values
        B = util.kernel_matrix(self.kernel, self.points, param)

        H, P = lanczos_decomposition(B, f, self.tol)
        Hinv = invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return util.loocv_error(coeffs, Binv)

    def __call__(self, x):
        return self.eval_func(
            self.kernel, self.points, self.alpha, self.beta, x, self.param
        )


class RBFInterpolationLinear(RBFInterpolationBase):
    def __init__(self, points, values, kernel, init_param, tol=1e-14):
        super().__init__(points, values, kernel, init_param)
        self.tol = tol

        self.coeffs = None

    def compute(self):
        if len(self.values.shape) > 1:
            self.coeffs = np.zeros(self.values.shape)

            for k in range(self.values.shape[1]):
                self.coeffs[:, k] = self.compute_coeffs(self.values[:, k])

            self.eval_func = linear_vector_eval

        else:
            self.coeffs = self.compute_coeffs(self.values)
            self.eval_func = linear_scalar_eval

    def compute_coeffs(self, f):
        B = util.kernel_matrix(self.kernel, self.points, self.param)

        H, P = lanczos_decomposition(B, f, self.tol)
        Hinv = invert_symm_tridiag(H)
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return coeffs

    def estimate_error(self, param):
        f = self.optimize_values
        B = util.kernel_matrix(self.kernel, self.points, param)

        H, P = lanczos_decomposition(B, f, self.tol)
        Hinv = invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return util.loocv_error(coeffs, Binv)

    def __call__(self, x):
        return self.eval_func(self.kernel, self.points, self.coeffs, x, self.param)


class RBFInterpolationAugmentedLinear(RBFInterpolationBase):
    def __init__(self, points, values, kernel, init_param, **kwargs):
        super().__init__(points, values, kernel, init_param)

        self.alpha = None
        self.beta = None

    def compute(self):
        if len(self.values.shape) > 1:
            self.coeffs = np.zeros(self.values.shape)

            for k in range(self.values.shape[1]):
                self.coeffs[:, k] = self.compute_coeffs(self.values[:, k])

            self.eval_func = linear_aug_vector_eval

        else:
            self.coeffs = self.compute_coeffs(self.values)
            self.eval_func = linear_aug_scalar_eval

    def compute_coeffs(self, f):
        B = util.augmented_kernel_matrix(self.kernel, self.points, self.param)
        n = f.shape[0]
        n_aug = B.shape[0]
        f_aug = np.hstack((f, np.zeros(n_aug - n)))

        coeffs = np.linalg.solve(B, f_aug)

        return coeffs

    def estimate_error(self, param):
        # TODO: augmented version
        f = self.optimize_values
        B = util.kernel_matrix(self.kernel, self.points, param)

        H, P = lanczos_decomposition(B, f, 1e-14)
        Hinv = invert_symm_tridiag(H)
        Binv = P @ Hinv @ P.T
        coeffs = (P @ Hinv[0, :]) * np.linalg.norm(f)

        return util.loocv_error(coeffs, Binv)

    def __call__(self, x):
        return self.eval_func(self.kernel, self.points, self.coeffs, x, self.param)


RBFInterpolation = RBFInterpolationRational
