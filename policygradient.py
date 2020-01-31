import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from utils import timeseries_plot_with_std_bands

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
    def __init__(self, env):
        self.env = env  # type: gym.Env
        self.episode_max_timesteps = 300
        self.learning_rate = 1e-3
        self.action_shape = self.env.action_space.n
        self.model = None  # type: tf.keras.Model
        self.setup_model()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def setup_model(self):
        # self.model.add(layers.Dense(64, activation="relu", input_shape=self.env.observation_space.shape))
        self.model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.env.observation_space.shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_shape)
        ])

    def learn(self, max_timesteps):
        steps = 0
        episodes = 1
        losses = []
        reward_sums = []
        episode_steps_list = []
        while steps < max_timesteps:
            observations, actions, rewards, episode_steps = self.collect_experience()
            steps += episode_steps
            returns = calculate_discounted_returns(rewards)
            # Convert observations, actions, returns to correctly shaped tensors or numpy arrays
            observations = tf.convert_to_tensor(np.stack(observations, axis=0), dtype='float32')
            actions = tf.stack(actions, axis=0)
            returns = tf.convert_to_tensor(np.array(returns), dtype='float32')
            ep_loss = self.training_step(observations, actions, returns)

            losses.append(ep_loss),
            reward_sums.append(np.sum(rewards))
            episode_steps_list.append(episode_steps)
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

    def test(self, n_episodes=10):
        episodes = 1
        reward_sums = []
        episode_steps_list = []
        while episodes <= n_episodes:
            observations, actions, rewards, episode_steps = self.collect_experience(render=True)
            reward_sums.append(np.sum(rewards))
            episode_steps_list.append(episode_steps)
            episodes += 1

        print("Test | Reward: {:>7.3f} | Steps: {:>5.1f} | Total steps:  {:>4d}".format(
            np.mean(reward_sums), np.mean(episode_steps_list), np.sum(episode_steps_list)))

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

        obs = env.reset()
        done = False
        episode_steps = 0
        while not done and episode_steps < self.episode_max_timesteps:
            if render:
                env.render()
            _observations.append(obs)  # record the observation here, because this way _actions[i] will be the one calculated from _observations[i]
            logprobactions = self.model(obs[None, ...])
            action = tf.random.categorical(logprobactions, num_samples=1)[0, 0]
            obs, reward, done, _ = env.step(action.numpy())

            _rewards.append(reward)
            _actions.append(action)
            episode_steps += 1

        return _observations, _actions, _rewards, episode_steps

    @tf.function(experimental_relax_shapes=True)
    def training_step(self, observations, actions, returns):
        # caluclate advantages for each step --> for now just normalize the returns
        scaled_returns = (returns - tf.math.reduce_mean(returns)) / tf.math.reduce_std(returns)

        # calculate loss (RL loss + value function loss)
        with tf.GradientTape() as tape:
            # Policy gradient loss: L = -log π(at|st) * A(at,st)
            logits = self.model(observations)
            logprobactions = tf.math.log(tf.keras.activations.softmax(logits))
            logprobat = tf.reduce_sum(tf.multiply(logprobactions, tf.one_hot(actions, depth=2)), axis=-1)
            loss = -tf.reduce_mean(tf.multiply(logprobat, scaled_returns))

            # SparseCategoricalCrossentropy can be used to validate the correctness of the above loss
            # print("L_PG: ", loss)
            # loss = cross_entropy_loss(actions, logits, sample_weight=scaled_returns[..., None])
            # print("L_CE: ", loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return tf.reduce_mean(loss)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    print(env)
    print(env.action_space.n, env.action_space, env.observation_space, env.observation_space.shape)
    agent = PolicyGradient(env)
    agent.learn(max_timesteps=50000)
    agent.test(10)


