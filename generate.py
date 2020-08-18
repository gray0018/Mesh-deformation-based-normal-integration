import numpy as np
from sklearn.preprocessing import normalize

n = np.random.rand(2, 2, 3)
n = normalize(n.reshape(-1,3)).reshape(n.shape)

np.save("big", n)
