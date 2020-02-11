import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from utils import *
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
        self.regularizer = None  # tf.keras.regularizers.l2(0.05)
        self.proba_distribution = None  # type: ProbaDistribution
        self.actor_model = self.setup_actor_model()  # type: tf.keras.Model
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.tensorboard_summary = tensorboard_setup()  # type: tf.summary.SummaryWriter

    def setup_actor_model(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.proba_distribution = Categorical(self.env.action_space)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.proba_distribution = DiagonalGaussian(self.env.action_space)
        actor_model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', kernel_regularizer=self.regularizer,
                         input_shape=self.env.observation_space.shape),
            layers.Dense(64, activation='relu', kernel_regularizer=self.regularizer),
            layers.Dense(self.proba_distribution.nn_feature_num)
        ]) #kernel_initializer=tf.keras.initializers.Zeros
        return actor_model

    def learn(self, max_timesteps, render_every_n_episode=30):
        steps = 0
        episodes = 1
        actor_losses = []
        critic_losses = []
        reward_sums = []
        episode_steps_list = []
        while steps < max_timesteps:
            if episodes % render_every_n_episode == 0:
                observations, actions, rewards, episode_steps, network_outputs = self.collect_experience(render=True)
            else:
                observations, actions, rewards, episode_steps, network_outputs = self.collect_experience(render=False)
            steps += episode_steps
            actions, observations, rewards, returns = self.prepare_data(actions, observations, rewards)
            if episodes == 1:
                tf.summary.trace_on(graph=True)
            actor_loss, critic_loss = self.training_step(observations, actions, rewards, returns, steps)
            if episodes == 1:
                with self.tensorboard_summary.as_default():
                    tf.summary.trace_export("Model_graph", step=0)

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            reward_sums.append(np.sum(rewards))
            episode_steps_list.append(episode_steps)
            with self.tensorboard_summary.as_default():
                tf.summary.scalar("Training/Episode reward sum", np.sum(rewards), step=steps)
                tf.summary.histogram("Training/Rewards", rewards, step=steps)
                tf.summary.histogram("Training/Returns", returns, step=steps)
            self.proba_distribution.log_histograms(actions, network_outputs, self.tensorboard_summary, steps)
            if episodes % 10 == 0:
                print("Episode {:>4d} | Reward: {:>7.3f} | Actor Loss: {:>8.4f} | Critic Loss: {:>8.4f} | "
                      "Steps: {:>4.1f} | Total steps:  {:>4d}".format(
                    episodes, np.mean(reward_sums[-10:]), np.mean(actor_losses[-10:]), np.mean(critic_losses[-10:]),
                    np.mean(episode_steps_list[-10:]), np.sum(episode_steps_list)))
            episodes += 1

        plt.subplot(311)
        timeseries_plot_with_std_bands(reward_sums, window_size=10, ylabel="Episode reward sum")
        plt.subplot(312)
        timeseries_plot_with_std_bands(actor_losses, window_size=10, ylabel="Actor loss", xlabel="Episodes") # xticks=np.cumsum(episode_steps_list)
        if not np.any(np.isnan(critic_losses)):
            plt.subplot(313)
            timeseries_plot_with_std_bands(critic_losses, window_size=10, ylabel="Critic loss", xlabel="Episodes")
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

        obs = self.env.reset()
        done = False
        episode_steps = 0
        while not done and episode_steps < self.episode_max_timesteps:
            if render:
                self.env.render()
            _observations.append(obs)  # record the observation here, because this way _actions[i] will be the one calculated from _observations[i]
            network_output = self.actor_model(obs[None, ...])
            action = self.proba_distribution.sample(network_output)
            obs, reward, done, _ = self.env.step(action.numpy())

            _rewards.append(reward)
            _actions.append(action)
            _network_outputs.append(network_output)
            episode_steps += 1

        return _observations, _actions, _rewards, episode_steps, _network_outputs

    def prepare_data(self, actions, observations, rewards):
        returns = calculate_discounted_returns(rewards)
        # Convert observations, actions, returns to correctly shaped tensors or numpy arrays
        observations = tf.convert_to_tensor(np.stack(observations, axis=0), dtype='float32')
        actions = tf.stack(actions, axis=0)
        returns = tf.convert_to_tensor(np.array(returns), dtype='float32')
        return actions, observations, rewards, returns

    def training_step(self, observations, actions, rewards, returns, steps):
        # caluclate advantages for each step --> for now just normalize the returns
        ep_loss, ep_gradnorm = self.training_step_actor(observations, actions, advantage_estimate=returns)
        with self.tensorboard_summary.as_default():
            tf.summary.scalar("Training/Actor loss", ep_loss, step=steps)
            tf.summary.scalar("Training/Actor Grad Norm", ep_gradnorm, step=steps)
        return ep_loss, np.nan

    @tf.function(experimental_relax_shapes=True)
    def training_step_actor(self, observations, actions, advantage_estimate):
        normalized_advantages = safe_normalize_tf(advantage_estimate)
        with tf.GradientTape() as tape:
            # Policy gradient loss: L = -log π(at|st) * A(at,st)
            network_output = self.actor_model(observations)
            logprobat = self.proba_distribution.neg_log_prob_a_t(network_output, actions)
            loss = tf.reduce_mean(tf.multiply(logprobat, normalized_advantages))

            # SparseCategoricalCrossentropy can be used to validate the correctness of the above loss for discrete actions
            # print("L_PG: ", loss)
            # loss = cross_entropy_loss(actions, network_output, sample_weight=normalized_advantages[..., None])
            # print("L_CE: ", loss)

        gradients = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor_model.trainable_variables))
        return tf.reduce_mean(loss), tf.linalg.global_norm(gradients)

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
    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLanderContinuous-v2')
    print(env)
    print("Action space: ", env.action_space, "\nObservation space:", env.observation_space)
    agent = PolicyGradient(env)
    agent.learn(max_timesteps=250000)
    agent.test(10)

