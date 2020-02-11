import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import tensorflow as tf


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
    plt.grid('on')
    if xticks is not None:
        x_tic_locs, _ = plt.xticks()
        plt.xticks(x_tic_locs[:-1], xticks[x_tic_locs[:-1].astype(int)])


def tensorboard_setup(base_path='tensorboard', run_subfolder="date-time"):
    if run_subfolder == "date-time":
        run_subfolder = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    tensorboard_root= os.path.join(base_path, run_subfolder)
    summary_writer = tf.summary.create_file_writer(tensorboard_root)
    return summary_writer


def history_dict_to_tensorboard(summary_writer, history_dict, step):
    with summary_writer.as_default():
        for key, value in history_dict.items():
            if value["type"] == "scalar":
                tf.summary.scalar(key, value["value"], step)
            elif value["type"] == "hist" or value["type"] == "histogram":
                tf.summary.histogram(key, value["value"], step)
            elif value["type"] == "img" or value["type"] == "image":
                tf. summary.image(key, value["value"], step)


def safe_normalize_tf(x: tf.Tensor):
    std = tf.math.reduce_std(x)
    if std > 0.:
        return (x - tf.math.reduce_mean(x)) / std
    else:
        # this most likely just returns zero
        return x - tf.math.reduce_mean(x)


if __name__ == '__main__':
    t = 100
    ts = np.random.random(t) + np.array(np.sin(t))
    plt.subplot(211)
    timeseries_plot_with_std_bands(ts, window_size=5)
    plt.subplot(212)
    timeseries_plot_with_std_bands(ts, window_size=15, xticks=-np.array(range(t+1)))
    plt.show()

