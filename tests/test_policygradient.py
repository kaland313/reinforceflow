import unittest
from unittest import TestCase
import numpy as np
from PolicyGradient import calculate_discounted_returns

# Credits: https://github.com/csxeba/Learn-TensorFlow-2.0-The-Hard-Way
def discount_rewards(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


class Test(TestCase):
    def test_calculate_discounted_returns(self):
        rewards = np.random.random(10)
        np.testing.assert_allclose(discount_rewards(rewards, gamma=0.99),
                                   calculate_discounted_returns(rewards, gamma=0.99))
        rewards = np.random.random(10000)*10000
        np.testing.assert_allclose(discount_rewards(rewards, gamma=0.99),
                                   calculate_discounted_returns(rewards, gamma=0.99))


if __name__ == '__main__':
    unittest.main()


