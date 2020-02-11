import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow_probability import distributions as tfd

mean_range = np.arange(-2.5, 2.5, 0.05)
sigma_range = np.arange(0.1, 2, 0.05)
mesh_mean, mesh_sigma = np.meshgrid(mean_range, sigma_range)
L=np.zeros_like(mesh_mean)
for i, row in enumerate(mesh_mean):
    for j, _ in enumerate(row):
        dist = tfd.MultivariateNormalDiag(loc=[mesh_mean[i, j]], scale_diag=[mesh_sigma[i, j]])
        neg_log_prob_a_t = -dist.log_prob([0.]).numpy()
        L[i, j] = neg_log_prob_a_t

L = np.clip(L, -10., 2.)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(mesh_mean, mesh_sigma, L, linewidth=0, cmap=cm.coolwarm)
ax.set_zlim3d(top=2.)
plt.xlabel("Mean")
plt.ylabel("Stdev")
plt.show()

