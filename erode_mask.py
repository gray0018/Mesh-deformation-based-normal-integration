import numpy as np

def erode_mask(mask):
    new_mask = np.zeros_like(mask)
    for i in range(1, mask.shape[0]-1):
        for j in range(1, mask.shape[1]-1):
            if mask[i+1,j] and mask[i-1,j] and mask[i,j-1] and mask[i,j+1]:
                new_mask[i, j] = 1
    return new_mask.astype(np.bool_)

