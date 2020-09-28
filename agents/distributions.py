import numpy as np
import tensorflow as tf
import gym.spaces
import tensorflow_probability as tfp


class ProbaDistribution:
    """Abstract class whose subclasses should implement the probabilistic outputs of reinforcement learning agents.
    Subclasses could be simply wrappers of tf and tfp distributions.
    """
    def __init__(self, nn_feature_num):
        self.nn_feature_num = nn_feature_num

    def sample(self, network_outputs) -> tf.Tensor:
        """Return samples from a distribution that is parameterised by network_outputs.
        E.g.: - network_outputs defines mean and variance of a gaussian distribution.
              - network_outputs defines class probabilities of a categorical distribution.
        """
        raise NotImplemented

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        """Return the negative *(-1) log probabilities of sampled actions, assuming that they were sampled form
        a distribution that is parameterised by network_outputs."""
        raise NotImplemented

    def prob_a_t(self, network_outputs, sampled_actions):
        """Return the probabilities of sampled actions, assuming that they were sampled form a distribution that
        is parameterised by network_outputs."""
        raise NotImplemented

    def kl(self, network_outputs_p, network_outputs_q):
        """Calculate the Kullback-Leibler divergence of p and q distributions, which are parameterised by
        network_outputs_p and network_outputs_q.
        p and q belong to the same distribution class (e.g. n class categorical), but with different parameters.
        """
        raise NotImplemented
    
    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        """Log histograms for the distribution in Tensorboard."""
        pass


class Categorical(ProbaDistribution):
    def __init__(self, action_space):
        super(Categorical, self).__init__(action_space.n)

    def sample(self, network_outputs):
        """
        :param network_outputs: Log-probabilities for all classes, 2-D Tensor with shape [batch_size, num_classes]
        :return: Scalar (Tensor with shape=()) or 1-D tensor, depending on batch_size
        Example:
            >>> import random
            >>> random.seed(1)
            >>> dist = Categorical(gym.spaces.Discrete(4))
            >>> sample = dist.sample(np.array([[0.1, 0.4, 0.2, 0.3]])).numpy()
            2
        """
        return tf.squeeze(tf.random.categorical(network_outputs, num_samples=1))

    def prob_a_t(self, network_outputs, sampled_actions):
        """
        :param network_outputs: Log-probabilities for all classes, 2-D Tensor with shape [batch_size, num_classes]
        :param sampled_actions: tf.Tensor or np.array of integers in the [0, num_classes] range
        :return: tf.Tensor with the same shame as sampled_actions
        Example:
            >>> dist = Categorical(gym.spaces.Discrete(4))
            >>> dist.prob_a_t(tf.convert_to_tensor(np.log([[0.1, 0.4, 0.2, 0.3]]), dtype=tf.float32), np.array([2]))
            0.2
        """
        probactions = tf.keras.activations.softmax(network_outputs)
        probat = tf.reduce_sum(tf.multiply(probactions, tf.one_hot(sampled_actions, depth=self.nn_feature_num)), axis=-1)
        return probat

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        return -tf.math.log(self.prob_a_t(network_outputs, sampled_actions))

    def kl(self, network_outputs_p, network_outputs_q):
        """
        :param network_outputs_p: Log-probabilities for all classes, 2-D Tensor with shape [batch_size, num_classes]
        :param network_outputs_q: Log-probabilities for all classes, 2-D Tensor with shape [batch_size, num_classes]
        """
        # network_outputs_p and network_outputs_q are logits, to get probability distribution (like) numbers use softmax
        p = tf.keras.activations.softmax(network_outputs_p)
        q = tf.keras.activations.softmax(network_outputs_q)
        return tf.reduce_sum(tf.multiply(p, tf.math.log(tf.math.divide(p, q))), axis=-1)


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
        mean_vector, std_vector, _ = self.split_network_features(network_outputs)
        dist = self.tfp_distribution(mean_vector, std_vector)
        action = dist.sample([1])
        action = tf.squeeze(action, axis=range(action.ndim-1))  # keep the last dim, squeeze all others
        action = tf.clip_by_value(action, self.action_space.low, self.action_space.high)
        return action

    def prob_a_t(self, network_outputs, sampled_actions):
        mean_vector, std_vector, log_std_vector = self.split_network_features(network_outputs)
        dist = self.tfp_distribution(mean_vector, std_vector)
        prob_a_t = dist.prob(sampled_actions)
        return prob_a_t

    def neg_log_prob_a_t(self, network_outputs, sampled_actions):
        mean_vector, std_vector, log_std_vector = self.split_network_features(network_outputs)
        dist = self.tfp_distribution(mean_vector, std_vector)
        log_prob_a_t = dist.log_prob(sampled_actions)
        return -log_prob_a_t

    def kl(self, network_outputs_p, network_outputs_q):
        mean_vector_p, std_vector_p, _ = self.split_network_features(network_outputs_p)
        p = self.tfp_distribution(mean_vector_p, std_vector_p)
        mean_vector_q, std_vector_q, _ = self.split_network_features(network_outputs_q)
        q = self.tfp_distribution(mean_vector_q, std_vector_q)
        return tfp.distributions.kl_divergence(p, q)

    @staticmethod
    def split_network_features(network_outputs):
        mean_vector, log_std_vector = tf.split(network_outputs, num_or_size_splits=2, axis=-1)
        std_vector = tf.exp(log_std_vector)
        return mean_vector, std_vector, log_std_vector
        
    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        with tensorboard_summary.as_default():
            tf.summary.histogram("Probabilistic Actions/Sampled actions", sampled_actions, step=step, )
            mean_vector, std_vector, _ = self.split_network_features(network_outputs)
            tf.summary.histogram("Probabilistic Actions/Predicted mean", mean_vector, step=step)
            tf.summary.histogram("Probabilistic Actions/Predicted std", std_vector, step=step)

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

    def kl(self, network_outputs_p, network_outputs_q):
        mean_vector_p = network_outputs_p
        std = tf.exp(self.log_std)
        p = self.tfp_distribution(mean_vector=mean_vector_p, std_vector=tf.ones_like(mean_vector_p[0]) * std)
        mean_vector_q = network_outputs_q
        # std = tf.exp(self.log_std)
        q = self.tfp_distribution(mean_vector=mean_vector_q, std_vector=tf.ones_like(mean_vector_p[0]) * std)
        return tfp.distributions.kl_divergence(p, q)

    def log_histograms(self, sampled_actions, network_outputs, tensorboard_summary, step):
        with tensorboard_summary.as_default():
            tf.summary.histogram("Probabilistic Actions/Sampled actions", sampled_actions, step=step)
            tf.summary.histogram("Probabilistic Actions/Predicted mean", network_outputs, step=step)
            tf.summary.scalar("Probabilistic Actions/Global action std", tf.exp(self.log_std), step=step)

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

