#%%
import numpy as np
import numba as nb

from pyrbfpu.common import *
from pyrbfpu.boxpartition import *

#%%
x, y = np.mgrid[0:1:50j, 0:1:50j]
x = x.flatten()
y = y.flatten()

# x, y = np.random.uniform(0, 1, (2, 50*50))

points = np.array([x, y]).T


def test_func(x, y):
    return np.sin(2.5 * x ** 2 + 6 * y ** 2) - np.sin(2 * x ** 2 + 4 * y - 0.5)


def test_func(x, y):
    return [
        np.sin(2.5 * x ** 2 + 6 * y ** 2) - np.sin(2 * x ** 2 + 4 * y - 0.5),
        np.sin(2.5 * x ** 2 + 6 * y ** 2),
        -np.sin(2 * x ** 2 + 4 * y - 0.5),
    ]


vals = test_func(x, y)
vals = np.array(vals).T

pu = RatRBFPartUnityInterpolation(points, vals, 100)
pu.domain_decomposition()

test_points = [
    points[211],
    points[523],
    [0.5, 0.5],
    [0.2, 0.6547],
    [0.2, 0.605],
    [0.6546, 0.31],
    [0.8, 0.6347],
    [0.9, 0.821],
    points[421] + 0.001,
]

for point in test_points:
    print("--- New Point: ", list(point))
    print("{}\n".format(pu(point) - test_func(*point)))

