import numpy as np
from scipy.linalg import ldl, solve_triangular
from scipy.sparse.linalg import eigsh, spsolve_triangular

import numba as nb

from pyrbfpu.util import lanczos_decomposition
from pyrbfpu.kernels import kernel_matrix


@nb.njit
def interp_eval(kernel, points, alpha, beta, x):
    dim = alpha.shape[1]
    num = np.zeros(dim)
    denom = np.zeros(dim)

    for i in range(points.shape[0]):
        phi = kernel(x, points[i])
        for k in range(dim):
            num[k] += alpha[i, k] * phi
            denom[k] += beta[i, k] * phi

    return num / denom


class VectorRationalRBF:
    def __init__(self, points, values, kernel, tol=1e-14):
        self.points = points
        self.values = values

        self.kernel = kernel
        self.tol = tol

        self.alpha = None
        self.beta = None

        self.eval_func = interp_eval

    def compute(self):
        f = self.values
        B = kernel_matrix(self.kernel, self.points)

        self.alpha = np.zeros(f.shape)
        self.beta = np.zeros(f.shape)

        for k in range(self.values.shape[1]):
            f = self.values[:, k]

            if np.allclose(f, 0.0, rtol=0.0, atol=1e-14):
                self.alpha[:, k] = 0.0
                self.beta[:, k] = 1.0

            H, P = lanczos_decomposition(B, f, self.tol)
            U, s, Vh = np.linalg.svd(H, full_matrices=False)

            c = P @ (Vh[0] / s[0])

            self.alpha[:, k] = P @ Vh.T @ ((U.T @ P.T @ (f * c)) / s)
            self.beta[:, k] = c / s[0]

    @property
    def computed(self):
        return self.eval_func is not None

    def __call__(self, x):
        return self.eval_func(self.kernel, self.points, self.alpha, self.beta, x)
