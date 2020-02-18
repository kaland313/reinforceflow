import unittest
from unittest import TestCase
import numpy as np
from utils.reward_calc import calculate_discounted_returns, calculate_generalized_advantage_estimate


# Credits: https://github.com/csxeba/trickster/blob/dev/trickster/processing/reward_shaping.py
def discount(rewards, dones, gamma=None):
    discounted = np.empty_like(rewards)
    cumulative_sum = 0.
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_sum *= (1 - dones[i]) * gamma
        cumulative_sum += rewards[i]
        discounted[i] = cumulative_sum
    return discounted

def compute_gae(rewards, values, values_next, dones, gamma, lmbda):
    delta = rewards + gamma * values_next * (1 - dones) - values
    advantages = discount(delta, dones, gamma=gamma * lmbda)
    returns = advantages + values
    return advantages, returns


class Test(TestCase):
    def test_calculate_discounted_returns(self):
        rewards = np.random.random(10)
        dones = np.zeros_like(rewards)
        dones[-1] = 1
        np.testing.assert_allclose(discount(rewards, dones, gamma=0.99),
                                   calculate_discounted_returns(rewards, dones, gamma=0.99))
        # rewards = np.random.random(10000)*10000
        # np.testing.assert_allclose(discount(rewards, dones, gamma=0.99),
        #                            calculate_discounted_returns(rewards, gamma=0.99))

    def test_calculate_gae(self):
        gamma = 0.99
        gae_lambda = 0.97
        size = 10
        rewards = np.random.random(size)
        values = np.random.random(size + 1)
        dones = np.zeros_like(rewards)
        dones[-1] = 1
        np.testing.assert_allclose(compute_gae(rewards, values[0:-1], values[1:], dones, gamma, gae_lambda),
                                   calculate_generalized_advantage_estimate(rewards, values[0:-1], dones, gae_lambda, gamma),
                                   rtol=1e-6)

if __name__ == '__main__':
    unittest.main()


