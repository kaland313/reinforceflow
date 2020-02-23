import numpy as np
import tensorflow as tf
import gym.spaces
import tensorflow_probability as tfp

class ProbaDistribution:
    def __init__(self, nn_feature_num):
        self.nn_feature_num = nn_feature_num

    def sample(self, network_outputs):
        NotImplemented

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        NotImplemented

    def prob_a_t(self, network_outputs, sampled_actions):
        NotImplemented
    
    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        NotImplemented
        
class Categorical(ProbaDistribution):
    def __init__(self, action_space):
        super(Categorical, self).__init__(action_space.n)

    def sample(self, network_outputs):
        return tf.random.categorical(network_outputs, num_samples=1)[0, 0]

    def prob_a_t(self, network_outputs, sampled_actions):
        probactions = tf.keras.activations.softmax(network_outputs)
        probat = tf.reduce_sum(tf.multiply(probactions, tf.one_hot(sampled_actions, depth=self.nn_feature_num)), axis=-1)
        return probat

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        return -tf.math.log(self.prob_a_t(network_outputs, sampled_actions))


class DiagonalGaussian(ProbaDistribution):
    def __init__(self, action_space, tanh_transform=True):
        # *2 to predict mean and std of the Gaussian distribution
        super(DiagonalGaussian, self).__init__(nn_feature_num=action_space.shape[0]*2)
        self.action_space = action_space  # type: gym.spaces.Box
        self.tanh_transform = tanh_transform
        if self.tanh_transform:
            self.transform_scale = (self.action_space.high - self.action_space.low) / 2.
            self.transform_shift = (self.action_space.high + self.action_space.low) / 2.

    def sample(self, network_outputs):
        mean_vector, std_vector, _ = self.split_network_feautres(network_outputs)
        dist = self.tfp_distribution(mean_vector, std_vector)
        action = dist.sample([1])
        action = tf.squeeze(action, axis=range(action.ndim-1))  # keep the last dim, squeeze all others
        action = tf.clip_by_value(action, self.action_space.low, self.action_space.high)
        return action

    def prob_a_t(self, network_outputs, sampled_actions):
        mean_vector, std_vector, log_std_vector = self.split_network_feautres(network_outputs)
        dist = self.tfp_distribution(mean_vector, std_vector)
        prob_a_t = dist.prob(sampled_actions)
        return prob_a_t

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        mean_vector, std_vector, log_std_vector = self.split_network_feautres(network_outputs)
        dist = self.tfp_distribution(mean_vector, std_vector)
        log_prob_a_t = dist.log_prob(sampled_actions)
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

    def tfp_distribution(self, mean_vector, std_vector):
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean_vector, scale_diag=std_vector)
        if self.tanh_transform:
            transforms = tfp.bijectors.Chain(bijectors=[tfp.bijectors.Tanh(),
                                                        tfp.bijectors.Affine(shift=self.transform_shift,
                                                                             scale_diag=self.transform_scale)])
            dist = tfp.distributions.TransformedDistribution(distribution=dist,
                                                             bijector=transforms,
                                                             name='TanhMultivariateNormalDiag')
        return dist


class DiagonalGaussianGlobalStd(ProbaDistribution):
    def __init__(self, action_space, initial_std=0.5, tanh_transform=True):
        super(DiagonalGaussianGlobalStd, self).__init__(action_space.shape[0])
        self.action_space = action_space  # type: gym.spaces.Box
        self.log_std = tf.Variable(np.log(initial_std), trainable=True, name="action_global_log_std", dtype='float32')
        self.tanh_transform = tanh_transform
        if self.tanh_transform:
            self.transform_scale = (self.action_space.high - self.action_space.low) / 2.
            self.transform_shift = (self.action_space.high + self.action_space.low) / 2.

    def sample(self, network_outputs):
        mean_vector = network_outputs
        random_normal = tf.random.normal(shape=mean_vector.shape)
        std = tf.exp(self.log_std)
        action = (random_normal * std + mean_vector)[0]
        # action = tf.clip_by_value(action, self.action_space.low, self.action_space.high)
        return action

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        mean_vector = network_outputs
        std = tf.exp(self.log_std)
        dist = self.tfp_distribution(mean_vector=mean_vector, std_vector=tf.ones_like(mean_vector[0])*std)
        log_prob_a_t = dist.log_prob(sampled_actions)
        return -log_prob_a_t

    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        with tensorboard_summary.as_default():
            tf.summary.histogram("Actions/Sampled actions", sampled_actions, step=step)
            tf.summary.histogram("Actions/Predicted mean", network_outputs, step=step)
            tf.summary.scalar("Actions/Global action std", tf.exp(self.log_std), step=step)

    def tfp_distribution(self, mean_vector, std_vector):
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean_vector, scale_diag=std_vector)
        if self.tanh_transform:
            transforms = tfp.bijectors.Chain(bijectors=[tfp.bijectors.Tanh(),
                                                        tfp.bijectors.Affine(shift=self.transform_shift,
                                                                             scale_diag=self.transform_scale)])
            dist = tfp.distributions.TransformedDistribution(distribution=dist,
                                                             bijector=transforms,
                                                             name='TanhMultivariateNormalDiag')
        return dist

