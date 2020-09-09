import numpy as np
from matplotlib import pyplot as plt

d = np.load("sphere_gt_depth.npy")

m, n = d.shape

mask = np.abs(d)<=10000

mask[:, :] = np.False_
mask[m//2, n//4] = np.True_
mask[m//2, n*3//4] = np.True_

d[~mask] = np.nan

d[m//2, n//4] *= 1.5
d[m//2, n*3//4] *= 0.5

np.save("sphere_sparse_depth", d)

