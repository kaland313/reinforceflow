import numpy as np
import tensorflow as tf


def safe_normalize_tf(x: tf.Tensor):
    std = tf.math.reduce_std(x)
    if std > 0.:
        return (x - tf.math.reduce_mean(x)) / std
    else:
        # this most likely just returns zero
        return x - tf.math.reduce_mean(x)


def calculate_discounted_returns(rewards, gamma=0.99):
    """Finite horizon discounted return"""
    returns = np.zeros_like(rewards)
    for i, _ in enumerate(returns):
        # We iterate through rewards in reverse, hence "-i", but the last element is indexed as -1
        returns[i] = rewards[-i - 1]
        if i != 0:
            # returns[i-1] is just the return at the next timesteps, which include all future rewards discounted
            returns[i] += returns[i - 1] * gamma
    return returns[::-1]


def calculate_generalized_advantage_estimate(rewards, values, dones, gae_lambda=0.97, discount_gamma=0.99):
    # delta_t = gamma*V(t+1) + r_t - V(t)
    Vt = values[:-1]   # the last element is only used as V(t+1), thus it's exlcuded from V(t)
    Vt1 = values[1:]   # the first element is only used as V(t), thus it's exlcuded from V(t+1)
    delta = discount_gamma * Vt1 * (1 - dones) + rewards - Vt
    advantage_estimates = calculate_discounted_returns(delta, gamma=gae_lambda * discount_gamma)
    advantage_estimates = tf.convert_to_tensor(advantage_estimates, dtype='float32')
    # A = V - Q ~= returns - values
    returns = advantage_estimates + Vt
    return advantage_estimates, returns


