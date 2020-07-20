import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

n = np.load("data/bunny_normal.npy")
sparse = np.load("data/bunny_sparse_depth.npy")
mask_bg = np.load("data/mask_bg.npy")

sparse[~np.isnan(sparse)] = 1
sparse[mask_bg] = 5

plt.style.use(['science','no-latex'])




fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True,sharey=True,figsize=(10,4.5))
plt.subplot(1,2,1),plt.imshow(n/2+.5)
plt.title('Normal map'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(sparse)
plt.title('Sparse Depth Prior'), plt.xticks([]), plt.yticks([])

red_patch = mpatches.Patch(color='#440154', label='Depth prior')
plt.legend(handles=[red_patch])

plt.show()