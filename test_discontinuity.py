import numpy as np
from matplotlib import pyplot as plt

from pyrbfpu.rbfpu import RBFUnityPartitionInterpolation

np.random.seed(12351)
points = np.linspace(-1, 1, 300)


def test_func(x):
    if x < 0.0:
        return -1.0 #+ np.random.normal(-1.0, 0.003)
    else:
        return 1.0 #+ x


vals = np.array([test_func(x) for x in points])

pu = RBFUnityPartitionInterpolation(
    points, vals, 100, tol=1e-14, rbf="buhmann_C3", box_overlap=0.01
)
sample_space = np.linspace(-1, 1, 1000)
interp_vals = np.array([pu(x) for x in sample_space])

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(points, vals, "+")
axes.plot(sample_space, interp_vals, "--")
plt.show()
