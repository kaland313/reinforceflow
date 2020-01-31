import numpy as np
import matplotlib.pyplot as plt

def timeseries_plot_with_std_bands(timeseries, window_size: int, xlabel=None, ylabel=None, xticks=None, color='C0'):
    _timeseries = np.array(timeseries)
    i = 0
    means = []
    stds = []
    while i + window_size <= _timeseries.shape[0]:
        means.append(np.mean(_timeseries[i: i + window_size]))
        stds.append(np.std(_timeseries[i: i + window_size]))
        i += 1

    means = np.array(means)
    stds = np.array(stds)
    t = range(window_size, _timeseries.shape[0] + 1)

    plt.plot(t, means, color)
    plt.fill_between(t, means+stds, means-stds, linewidth=0., color=color, alpha=0.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        x_tic_locs, _ = plt.xticks()
        plt.xticks(x_tic_locs[:-1], xticks[x_tic_locs[:-1].astype(int)])


if __name__ == '__main__':
    t = 100
    ts = np.random.random(t) + np.array(np.sin(t))
    plt.subplot(211)
    timeseries_plot_with_std_bands(ts, window_size=5)
    plt.subplot(212)
    timeseries_plot_with_std_bands(ts, window_size=15, xticks=-np.array(range(t+1)))
    plt.show()
