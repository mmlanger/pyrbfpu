import time

import numpy as np
from matplotlib import pyplot as plt

from pyrbfpu.common import *


np.random.seed(12351)
points = np.random.uniform(2, 3, (300, 2))


def test_func(point):
    x, y = point
    r = np.sqrt(x ** 2 + y ** 2)
    return np.sin(x) * np.cos(y) * (10 - r)


kernel = generate_kernel(inverse_multiquadric)
vals = np.array([test_func(x) for x in points])

rbf = RationalRBF(points, vals, kernel, 1.0, 1e-12)
rbf.estimate_error(0.1)

start = time.perf_counter()
rbf.optimize_param()
end = time.perf_counter()
print("TIME {}".format(end - start))
# print("TIME {} with EVALS {}".format(end - start, rbf.counter))

# start = time.perf_counter()
# for i in range(50):
#     rbf.estimate_error(0.1)
# end = time.perf_counter()
# print("TIME ", end - start)

eps_space = np.linspace(0.01, 3.5, 200)
errors = np.array([rbf.estimate_error(eps) for eps in eps_space])

fig = plt.figure()
axes = fig.add_subplot(111)
axes.axvline(rbf.param)
axes.plot(eps_space, errors, "+")
plt.show()
