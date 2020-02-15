import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow_probability as tfp

mean_range = np.arange(-2.5, 2.5, 0.1)
sigma_range = np.arange(0.1, 2, 0.1)
mesh_mean, mesh_sigma = np.meshgrid(mean_range, sigma_range)
L=np.zeros_like(mesh_mean)
for i, row in enumerate(mesh_mean):
    for j, _ in enumerate(row):
        dist = tfp.distributions.MultivariateNormalDiag(loc=[mesh_mean[i, j]], scale_diag=[mesh_sigma[i, j]])
        dist = tfp.distributions.TransformedDistribution(distribution=dist,
                                                         bijector=tfp.bijectors.Tanh(),
                                                         name='TanhMultivariateNormalDiag')
        neg_log_prob_a_t = -dist.log_prob([0.9]).numpy()
        L[i, j] = neg_log_prob_a_t

L = np.clip(L, -10., 2.)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(mesh_mean, mesh_sigma, L, linewidth=0, cmap=cm.coolwarm)
ax.set_zlim3d(top=2.)
plt.xlabel("Mean")
plt.ylabel("Stdev")
plt.show()

