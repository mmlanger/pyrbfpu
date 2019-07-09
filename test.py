import numpy as np
from pyrbfpu.common import *


np.random.seed(12351)
points = np.random.normal(0, 3, (4000, 2))
test_set = np.random.uniform(-7, 7, (25, 2))
sort_idx = np.linalg.norm(test_set, axis=1).argsort()
test_set = test_set[sort_idx]
tt = np.linalg.norm(test_set, axis=1)


def test_func(point):
    x, y = point
    r = np.sqrt(x**2 + y**2)
    return np.sin(x) * np.cos(y) * (10 - r)


vals = np.array([test_func(x) for x in points])

pu = RatRBFPartUnityInterpolation(points, vals, 100)

for point in [points[211], points[523]]:
    error = pu(point) - test_func(point)
    print("--- point {} with error {}".format(list(point), error))

for point in test_set:
    error = pu(point) - test_func(point)
    print("--- point {} with error {}".format(list(point), error))


# from mayavi import mlab

# mlab.points3d(*points.T, vals, vals, mode="point")
# # mesh = mlab.pipeline.delaunay2d(pts)
# # surf = mlab.pipeline.surface(mesh)

# x, y = np.mgrid[-6:6:0.1, -6:6:0.1]
# vals = np.zeros(x.shape)
# for i in range(vals.shape[0]):
#     for k in range(vals.shape[1]):
#         vals[i,k] = pu((x[i, k], y[i, k]))

# s = mlab.surf(x, y, vals)
# mlab.show()