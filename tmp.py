import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

def erode_mask(mask):
    new_mask = np.zeros_like(mask)
    for i in range(1, mask.shape[0]-1):
        for j in range(1, mask.shape[1]-1):
            if mask[i+1,j] and mask[i-1,j] and mask[i,j-1] and mask[i,j+1]:
                new_mask[i, j] = 1
    return new_mask.astype(np.bool_)

gt = np.load("data/bunny_depth.npy")
sparse = np.load("data/bunny_sparse_depth.npy")
res = np.load("data/output_depth.npy")
res_wodepthprior = np.load("data/output_depth_wodepthprior.npy")


mask_bg = np.load("data/mask_bg.npy")

plt.style.use(['science','no-latex'])

res_w_prior = np.abs(gt-np.nanmean(gt)-res+np.nanmean(res))
res_wo_prior = np.abs(gt-np.nanmean(gt)-res_wodepthprior+np.nanmean(res_wodepthprior))

vmin = min(np.nanmin(res_w_prior), np.nanmin(res_wo_prior))
vmax = max(np.nanmax(res_w_prior), np.nanmax(res_wo_prior))

fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True,sharey=True,figsize=(10,4.5))
plt.subplot(1,2,1),plt.imshow(res_wo_prior,vmin=vmin,vmax=vmax)
plt.title('W/O Depth Prior'), plt.xticks([]), plt.yticks([]), plt.xlabel("MAE={0}".format(np.nanmean(res_wo_prior)))
plt.subplot(1,2,2),plt.imshow(res_w_prior,vmin=vmin,vmax=vmax)
plt.title('With Depth Prior'), plt.xticks([]), plt.yticks([]), plt.xlabel("MAE={0}".format(np.nanmean(res_w_prior)))
# plt.colorbar(fraction=0.046, pad=0.04)

#カラーバーの設定
axpos = axes[1].get_position()
cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
norm = colors.Normalize(vmin=vmin,vmax=vmax)
mappable = ScalarMappable(norm=norm)
mappable._A = []
fig.colorbar(mappable, cax=cbar_ax)

#余白の調整
plt.subplots_adjust(right=0.85)
plt.subplots_adjust(wspace=0.1)

plt.show()