import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable


if __name__ == '__main__':

    gt = np.load(sys.argv[1]) # ground truth depth
    est = np.load(sys.argv[2]) # estimated depth
    mask_bg = np.isnan(gt)

    plt.style.use(['science','no-latex'])

    gt = gt-np.nanmean(gt) # substract offset
    est = est-np.nanmean(est) # substract offset

    error_map = np.abs(gt-est)
 
    fig, axes = plt.subplots(figsize=(4.5,4.5))
    axes.imshow(error_map)
    axes.set_title('Error Map')
    axes.set_xlabel('MAE:{:0.2f}'.format(np.nanmean(error_map)))
    axes.set_xticks([]), axes.set_yticks([]) # x, y ticks off

    plt.show()

