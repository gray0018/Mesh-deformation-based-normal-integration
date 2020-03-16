from DGP import DGP
from matplotlib import pyplot as plt

path = "normal.png"
task = DGP(path)

for i in range(500):
    task.DGP_iter()

plt.figure()
plt.imshow(task.vertices_depth,"gray")
plt.show()
