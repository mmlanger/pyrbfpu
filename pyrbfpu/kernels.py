import numpy as np
import numba as nb

from pyrbfpu.util import dist


@nb.njit
def wendland(r):
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
    r_cubic = r_sqr * r
    return base_quad * base_quad * (32.0 * r_cubic + 25.0 * r_sqr + 8.0 * r + 1.0)


@nb.njit
def gauss(r):
    return np.exp(-r * r)


@nb.njit
def imq(r):
    return 1.0 / np.sqrt(1.0 + r * r)


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


def generate_kernel(kernel_func):
    @nb.njit
    def kernel(x1, x2, eps):
        return kernel_func(eps * dist(x1, x2))

    return kernel


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
