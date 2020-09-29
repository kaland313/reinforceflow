from unittest import TestCase
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
from ..agents.distributions import Categorical


class TestCategorical(TestCase):

    def test_prob_a_t(self):
        dist = Categorical(gym.spaces.Discrete(4))
        prob = dist.prob_a_t(tf.convert_to_tensor(np.log([[0.1, 0.4, 0.2, 0.3]]), dtype=tf.float32), np.array([2]))
        self.assertEqual(0.2, prob)

    def test_kl(self):
        dist = Categorical(gym.spaces.Discrete(4))
        my_kl = dist.kl(tf.convert_to_tensor(np.log([[0.1, 0.4, 0.2, 0.3]]), dtype=tf.float32),
                        tf.convert_to_tensor(np.log([[0.1, 0.4, 0.2, 0.3]]), dtype=tf.float32)).numpy()[0]
        self.assertEqual(0.0, my_kl)

        my_kl = dist.kl(tf.convert_to_tensor(np.log([[0.1, 0.4, 0.2, 0.3]]), dtype=tf.float32),
                        tf.convert_to_tensor(np.log([[0.7, 0.1, 0.1, 0.1]]), dtype=tf.float32)).numpy()[0]
        p = tfp.distributions.Categorical(probs=[0.1, 0.4, 0.2, 0.3])
        q = tfp.distributions.Categorical(probs=[0.7, 0.1, 0.1, 0.1])
        self.assertEqual(tfp.distributions.kl_divergence(p, q), my_kl)
