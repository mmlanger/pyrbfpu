import numpy as np
from matplotlib import pyplot as plt

from pyrbfpu.rbfpu import RatRBFPartUnityInterpolation

np.random.seed(12351)
points = np.linspace(-1, 1, 200)


def test_func(x):
    if x < 0.0:
        return -1.0
    else:
        return 1.0


vals = np.array([test_func(x) for x in points])

pu = RatRBFPartUnityInterpolation(points, vals, 50, tol=1e-14, rbf='matern_basic')
sample_space = np.linspace(-1, 1, 500)
interp_vals = np.array([pu(np.array([x])) for x in sample_space])

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(points, vals, "+")
axes.plot(sample_space, interp_vals, "--")
plt.show()
