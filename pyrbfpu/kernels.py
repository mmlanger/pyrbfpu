import numpy as np
import numba as nb

from pyrbfpu.util import dist


@nb.njit
def linear(r):
    return r


@nb.njit
def cubic(r):
    return r * r * r


@nb.njit
def quintic(r):
    r_sqr = r * r
    return r_sqr * r_sqr * r


@nb.njit
def thin_plate(r):
    if r == 0.0:
        return 0.0
    return r * r * np.log(r)


@nb.njit
def gaussian(r):
    return np.exp(-r * r)


@nb.njit
def multiquadric(r):
    return np.sqrt(1.0 + r * r)


@nb.njit
def inverse_multiquadric(r):
    return 1.0 / np.sqrt(1.0 + r * r)


@nb.njit
def matern_basic(r):
    return np.exp(-r)


@nb.njit
def matern_linear(r):
    return np.exp(-r) * (1.0 + r)


@nb.njit
def matern_quadratic(r):
    return np.exp(-r) * (3.0 + 3.0 * r + r * r)


@nb.njit
def matern_cubic(r):
    return np.exp(-r) * (15.0 + 15.0 * r + 6.0 * r * r + r * r * r)


@nb.njit
def wendland_C0(r):
    base = max(1.0 - r, 0.0)
    return base * base


@nb.njit
def wendland_C2(r):
    base = max(1.0 - r, 0.0)
    if base == 0.0:
        return 0.0

    base_sqr = base * base
    return base_sqr * base_sqr * (4.0 * r + 1.0)


@nb.njit
def wendland_C4(r):
    base = max(1.0 - r, 0.0)
    if base == 0.0:
        return 0.0

    base_sqr = base * base
    base_quad = base_sqr * base_sqr
    r_sqr = r * r
    return base_quad * base_sqr * (35.0 * r_sqr + 18.0 * r + 3.0)


@nb.njit
def wendland_C6(r):
    base = max(1.0 - r, 0.0)
    if base == 0.0:
        return 0.0

    base_sqr = base * base
    base_quad = base_sqr * base_sqr
    r_sqr = r * r
    r_cubic = r_sqr * r
    return base_quad * base_quad * (32.0 * r_cubic + 25.0 * r_sqr + 8.0 * r + 1.0)


@nb.njit
def buhmann_C2(r):
    r_sqr = r * r
    r_cubic = r_sqr * r
    r_quad = r_sqr * r_sqr
    result = (
        2.0 * r_quad * np.log(r)
        - 7.0 / 2.0 * r_quad
        + 16.0 / 3.0 * r_cubic
        - 2.0 * r_sqr
        + 1.0 / 6.0
    )
    return result


@nb.njit
def buhmann_C3(r):
    r_sqr = r * r
    r_quad = r_sqr * r_sqr
    r_pow35 = r ** 3.5
    result = (
        112.0 / 45.0 * r_pow35 * r
        + 16.0 / 3.0 * r_pow35
        - 7.0 * r_quad
        - 14.0 / 15.0 * r_sqr
        + 1.0 / 9.0
    )
    return result


def generate_kernel(kernel_func):
    @nb.njit
    def kernel(x1, x2, eps):
        return kernel_func(eps * dist(x1, x2))

    return kernel


def generate_hybrid_kernel(kernel_func_alpha, kernel_func_beta, gamma):
    @nb.njit
    def kernel(x1, x2, eps):
        d = dist(x1, x2)
        k_alpha = kernel_func_alpha(eps * d)
        k_beta = kernel_func_beta(eps * d)
        return k_alpha + gamma * k_beta

    return kernel


def generate_scale_func_v1(r, center):
    @nb.njit
    def scale_func(x):
        diff = x - center
        norm_sqr = np.dot(diff, diff)
        return 0.5 + np.sqrt(r * r - norm_sqr)

    return scale_func


def generate_scale_func_v2(u):
    @nb.njit
    def scale_func(x):
        return u * np.linalg.norm(x)

    return scale_func


def generate_scale_func_v3(r, center):
    @nb.njit
    def scale_func(x):
        diff = x - center
        norm_sqr = np.dot(diff, diff)
        return np.sqrt(r * r - norm_sqr)

    return scale_func


def generate_vskernel(kernel_func, scale_func, return_vsdist=False):
    @nb.njit
    def vsdist(x1, x2, param):
        n = x1.shape[0]

        r_sqr = 0.0
        for i in range(n):
            diff = x1[i] - x2[i]
            r_sqr += diff * diff

        diff = scale_func(x1, param) - scale_func(x2, param)
        r_sqr += diff * diff
        return np.sqrt(r_sqr)

    @nb.njit
    def vskernel(x1, x2, param):
        r = vsdist(x1, x2, param)
        return kernel_func(r)

    if return_vsdist:
        return vsdist, vskernel
    else:
        return vskernel
