import numpy as np
import tensorflow as tf
import gym.spaces
from tensorflow_probability import distributions as tfd

class ProbaDistribution:
    def __init__(self, N):
        self.nn_feature_num = N

    def sample(self, network_outputs):
        NotImplemented

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        NotImplemented
    
    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        NotImplemented
        
class Categorical(ProbaDistribution):
    def __init__(self, action_space):
        super(Categorical, self).__init__(action_space.n)

    def sample(self, network_outputs):
        return tf.random.categorical(network_outputs, num_samples=1)[0, 0]

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        logprobactions = tf.math.log(tf.keras.activations.softmax(network_outputs))
        logprobat = tf.reduce_sum(tf.multiply(logprobactions, tf.one_hot(sampled_actions, depth=self.nn_feature_num)), axis=-1)
        return -logprobat


class DiagonalGaussian(ProbaDistribution):
    def __init__(self, action_space):
        # *2 to predict mean and std of the Gaussian distribution
        super(DiagonalGaussian, self).__init__(action_space.shape[0]*2)
        self.action_space = action_space  # type: gym.spaces.Box

    def sample(self, network_outputs):
        mean_vector, std_vector, _ = self.split_network_feautres(network_outputs)
        random_normal = tf.random.normal(shape=mean_vector.shape)
        action = (random_normal*std_vector + mean_vector)[0]
        return tf.clip_by_value(action, self.action_space.low, self.action_space.high)

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        mean_vector, std_vector, log_std_vector = self.split_network_feautres(network_outputs)

        # The formula below can be derived by calculating the -log of the normal distribution
        # Actual implementation is coped from:
        # https://github.com/hill-a/stable-baselines/blob/c6acd1e6dcf40a824e4765198b705db7e5d7188e/stable_baselines/common/distributions.py#L402
        # neg_log_prob_a_t = 0.5 * tf.reduce_sum(tf.square((sampled_actions - mean_vector) / std_vector), axis=-1) \
        #                    + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(sampled_actions)[-1], tf.float32) \
        #                    + tf.reduce_sum(log_std_vector, axis=-1)

        dist = tfd.MultivariateNormalDiag(loc=mean_vector, scale_diag=std_vector)
        log_prob_a_t = dist.log_prob(sampled_actions)
        # np.testing.assert_allclose(neg_log_prob_a_t.numpy(), -log_prob_a_t, rtol=1e-6)

        return -log_prob_a_t
    
    @staticmethod
    def split_network_feautres(network_outputs):
        mean_vector, log_std_vector = tf.split(network_outputs, num_or_size_splits=2, axis=-1)
        std_vector = tf.exp(log_std_vector)
        return mean_vector, std_vector, log_std_vector
        
    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        with tensorboard_summary.as_default():
            tf.summary.histogram("Actions/Sampled actions", sampled_actions, step=step, )
            mean_vector, std_vector, _ = self.split_network_feautres(network_outputs)
            tf.summary.histogram("Actions/Predicted mean", mean_vector, step=step)
            tf.summary.histogram("Actions/Predicted std", std_vector, step=step)

