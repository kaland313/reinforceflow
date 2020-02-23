import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# import tensorflow as tf
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


fig = plt.figure()
lgnd = []
mean_range = np.arange(0, 1.5, 0.2)
sigma_range = np.arange(0.1, 2.5, 0.1)
minvals = []
minargs = []
for mean in mean_range:
    lgnd.append("{:0.3f}".format(mean))
    L = np.zeros_like(sigma_range)
    for j, sigma in enumerate(sigma_range):
        dist = tfp.distributions.MultivariateNormalDiag(loc=[mean], scale_diag=[sigma])
        dist = tfp.distributions.TransformedDistribution(distribution=dist,
                                                         bijector=tfp.bijectors.Tanh(),
                                                         name='TanhMultivariateNormalDiag')
        neg_log_prob_a_t = -dist.log_prob([0.]).numpy()
        L[j] = neg_log_prob_a_t
    plt.plot(sigma_range, L)
    minargs.append(sigma_range[np.argmin(L)])
    minvals.append(np.amin(L))
plt.legend(lgnd)
plt.ylim([-2., 3.])
plt.xlabel("Stdev")
plt.ylabel("-log prob Gaussian(0.)")
plt.title("-log prob Gaussian(0.) as the function of \n Mean and Std of the probability distribution")
plt.plot(minargs, minvals)
plt.show()


fig = plt.figure()
sigma_range = np.arange(0.1, 1.5, 0.2)
sample_range = np.arange(-2., 2., 0.01)
lgnd = []
for sigma in sigma_range:
    lgnd.append("σ={:0.3f}".format(sigma))
    dist = tfp.distributions.MultivariateNormalDiag(loc=[0.], scale_diag=[sigma])
    dist = tfp.distributions.TransformedDistribution(distribution=dist,
                                                     bijector=tfp.bijectors.Tanh(),
                                                     name='TanhMultivariateNormalDiag')
    pdf = np.zeros_like(sample_range)
    for i, sample in enumerate(sample_range):
        pdf[i] = dist.prob([sample]).numpy()
    plt.plot(sample_range, pdf)

plt.legend(lgnd)
plt.title("Probability density function for tanh gaussian distribution")
plt.show()



