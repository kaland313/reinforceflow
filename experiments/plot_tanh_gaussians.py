import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def gaussian(x, mu, sig):
    exponent = (x-mu)/sig
    return 1. / (sig * np.sqrt(2*np.pi)) * np.exp(-0.5 * exponent * exponent)


fig = plt.figure()
sigma_range = np.arange(0.1, 1.5, 0.4)
sample_range = np.arange(-3., 3., 0.05)
for sigma in sample_range:
    samples = np.random.normal()

plt.show()