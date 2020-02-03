import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from utils import timeseries_plot_with_std_bands, tensorboard_setup
from distributions import ProbaDistribution, Categorical, DiagonalGaussian


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


cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class PolicyGradient:
    "Implements the (vanilla) Policy Gradient algorithm"
    def __init__(self, env, episode_max_timesteps=300, learning_rate=1e-3):
        self.env = env  # type: gym.Env
        self.episode_max_timesteps = episode_max_timesteps
        self.learning_rate = learning_rate
        self.proba_distribution = None  # type: ProbaDistribution
        self.model = None  # type: tf.keras.Model
        self.setup_model()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.tensorboard_summary = tensorboard_setup()  # type: tf.summary.SummaryWriter
        self.regularizer = None  # tf.keras.regularizers.l2(0.05)

    def setup_model(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.proba_distribution = Categorical(self.env.action_space)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.proba_distribution = DiagonalGaussian(self.env.action_space)
        self.model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', activity_regularizer=self.regularizer,
                         input_shape=self.env.observation_space.shape),
            layers.Dense(64, activation='relu', activity_regularizer=self.regularizer),
            layers.Dense(self.proba_distribution.nn_feature_num, activity_regularizer=self.regularizer,
                         kernel_initializer=tf.keras.initializers.Zeros)
        ])

    def learn(self, max_timesteps):
        steps = 0
        episodes = 1
        losses = []
        reward_sums = []
        episode_steps_list = []
        while steps < max_timesteps:
            observations, actions, rewards, episode_steps, network_outputs = self.collect_experience()
            steps += episode_steps
            returns = calculate_discounted_returns(rewards)
            # Convert observations, actions, returns to correctly shaped tensors or numpy arrays
            observations = tf.convert_to_tensor(np.stack(observations, axis=0), dtype='float32')
            actions = tf.stack(actions, axis=0)
            returns = tf.convert_to_tensor(np.array(returns), dtype='float32')
            if episodes == 1:
                tf.summary.trace_on(graph=True)
            ep_loss = self.training_step(observations, actions, returns)
            if episodes == 1:
                with self.tensorboard_summary.as_default():
                    tf.summary.trace_export("Model_graph", step=0)

            losses.append(ep_loss),
            reward_sums.append(np.sum(rewards))
            episode_steps_list.append(episode_steps)
            with self.tensorboard_summary.as_default():
                tf.summary.scalar("Training/Episode reward sum", reward_sums[-1], step=steps)
                tf.summary.scalar("Training/Episode loss", ep_loss, step=steps)
            self.proba_distribution.log_histograms(actions, network_outputs, self.tensorboard_summary, steps)
            if episodes % 10 == 0:
                print("Episode {:>4d} | Reward: {:>7.3f} | Loss: {:>8.4f} | Steps: {:>4.1f} | Total steps:  {:>4d}".format(
                    episodes, np.mean(reward_sums[-10:]), np.mean(losses[-10:]),
                    np.mean(episode_steps_list[-10:]), np.sum(episode_steps_list)))
            episodes += 1

        plt.subplot(211)
        timeseries_plot_with_std_bands(reward_sums, window_size=10, ylabel="Episode reward sum")
        plt.subplot(212)
        timeseries_plot_with_std_bands(losses, window_size=10, ylabel="Episode loss", xlabel="Episodes") # xticks=np.cumsum(episode_steps_list)
        plt.show()

    def collect_experience(self, render=False):
        """
        :return: _observations, _actions, _rewards
         _observations: list of numpy arrays
         _actions: list of tf.Tensors with shape: ()
         _rewards: list of floats
        """
        _observations = []  # list of observations over the episode
        _rewards = []  # list of rewards over the episode
        _actions = []  # list of actions over the episode
        _network_outputs = []

        obs = env.reset()
        done = False
        episode_steps = 0
        while not done and episode_steps < self.episode_max_timesteps:
            if render:
                env.render()
            _observations.append(obs)  # record the observation here, because this way _actions[i] will be the one calculated from _observations[i]
            logits = self.model(obs[None, ...])
            action = self.proba_distribution.sample(logits)
            obs, reward, done, _ = env.step(action.numpy())

            _rewards.append(reward)
            _actions.append(action)
            _network_outputs.append(logits)
            episode_steps += 1

        return _observations, _actions, _rewards, episode_steps, _network_outputs

    @tf.function(experimental_relax_shapes=True)
    def training_step(self, observations, actions, returns):
        # caluclate advantages for each step --> for now just normalize the returns
        scaled_returns = (returns - tf.math.reduce_mean(returns)) / tf.math.reduce_std(returns)

        # calculate loss (RL loss + value function loss)
        with tf.GradientTape() as tape:
            # Policy gradient loss: L = -log π(at|st) * A(at,st)
            logits = self.model(observations)
            logprobat = self.proba_distribution.neg_log_prob_a_t(logits, actions)
            loss = tf.reduce_mean(tf.multiply(logprobat, scaled_returns))

            # SparseCategoricalCrossentropy can be used to validate the correctness of the above loss for discrete actions
            # print("L_PG: ", loss)
            # loss = cross_entropy_loss(actions, logits, sample_weight=scaled_returns[..., None])
            # print("L_CE: ", loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return tf.reduce_mean(loss)

    def test(self, n_episodes=10):
        episodes = 1
        reward_sums = []
        episode_steps_list = []
        while episodes <= n_episodes:
            observations, actions, rewards, episode_steps, network_outputs = self.collect_experience(render=True)
            reward_sums.append(np.sum(rewards))
            episode_steps_list.append(episode_steps)
            episodes += 1

        print("Test | Reward: {:>7.3f} | Steps: {:>5.1f} | Total steps:  {:>4d}".format(
            np.mean(reward_sums), np.mean(episode_steps_list), np.sum(episode_steps_list)))


if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v0')
    # env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLanderContinuous-v2')
    print(env)
    print("Action space: ", env.action_space, "\nObservation space:", env.observation_space)
    agent = PolicyGradient(env)
    agent.learn(max_timesteps=50000)
    agent.test(10)


