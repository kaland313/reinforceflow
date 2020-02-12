import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from distributions import ProbaDistribution, Categorical, DiagonalGaussian, DiagonalGaussianGlobalStd
from utils.reward_calc import calculate_discounted_returns, safe_normalize_tf
from utils.logging import tensorboard_setup, timeseries_plot_with_std_bands

cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class PolicyGradient:
    "Implements the (vanilla) Policy Gradient algorithm"
    def __init__(self,
                 env,
                 learning_rate=1e-3,
                 discount_gamma=0.99,
                 global_sigma_for_cont_action=False):
        self.env = env  # type: gym.Env
        self.learning_rate = learning_rate
        self.discount_gamma = discount_gamma

        self.regularizer = None  # tf.keras.regularizers.l2(0.05)
        self.global_sigma_for_cont_action = global_sigma_for_cont_action
        self.proba_distribution = None  # type: ProbaDistribution
        self.actor_model = None  # type: tf.keras.Model
        self.actor_trainable_vars = None

        self.setup_actor_model()
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.tensorboard_summary = tensorboard_setup()  # type: tf.summary.SummaryWriter


    def setup_actor_model(self):

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.proba_distribution = Categorical(self.env.action_space)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            if self.global_sigma_for_cont_action:
                self.proba_distribution = DiagonalGaussianGlobalStd(self.env.action_space)
            else:
                self.proba_distribution = DiagonalGaussian(self.env.action_space)
        self.actor_model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', kernel_regularizer=self.regularizer,
                         input_shape=self.env.observation_space.shape),
            layers.Dense(64, activation='relu', kernel_regularizer=self.regularizer),
            layers.Dense(self.proba_distribution.nn_feature_num)
        ]) #kernel_initializer=tf.keras.initializers.Zeros

        self.actor_trainable_vars = self.actor_model.trainable_variables
        if isinstance(self.proba_distribution, DiagonalGaussianGlobalStd):
            self.actor_trainable_vars += [self.proba_distribution.log_std]

    def learn(self, max_timesteps, render_every_n_episode=30):
        steps = 0
        episodes = 1
        actor_losses = []
        critic_losses = []
        reward_sums = []
        episode_steps_list = []
        while steps < max_timesteps:
            if episodes % render_every_n_episode == 0:
                observations, actions, rewards, dones, episode_steps, network_outputs = self.collect_experience(render=True)
            else:
                observations, actions, rewards, dones, episode_steps, network_outputs = self.collect_experience(render=False)
            steps += episode_steps
            actions, observations, rewards, dones = self.prepare_data(actions, observations, rewards, dones)
            if episodes == 1:
                tf.summary.trace_on(graph=True)
            actor_loss, critic_loss = self.training_step(observations, actions, rewards, dones, steps)
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
        _dones = []
        _network_outputs = []

        obs = self.env.reset()
        done = False
        episode_steps = 0
        while not done:
            if render:
                self.env.render()
            _observations.append(obs)  # record the observation here, because this way _actions[i] will be the one calculated from _observations[i]
            network_output = self.actor_model(obs[None, ...])
            action = self.proba_distribution.sample(network_output)
            obs, reward, done, _ = self.env.step(action.numpy())

            _rewards.append(reward)
            _actions.append(action)
            _dones.append(done)
            _network_outputs.append(network_output)
            episode_steps += 1

        # Append the last observation to be able to calculate V(t+1) when using the generalized advantage estimate
        _observations.append(obs)
        return _observations, _actions, _rewards, _dones, episode_steps, _network_outputs

    def prepare_data(self, actions, observations, rewards, dones):
        # Convert observations, actions, returns to correctly shaped tensors or numpy arrays
        observations = tf.convert_to_tensor(np.stack(observations, axis=0), dtype='float32')
        actions = tf.cast(tf.stack(actions, axis=0), dtype='float32')
        rewards = tf.cast(tf.stack(rewards, axis=0), dtype='float32')
        dones = tf.cast(tf.stack(dones, axis=0), dtype='float32')
        return actions, observations, rewards, dones

    def training_step(self, observations, actions, rewards, dones, steps):
        returns = calculate_discounted_returns(rewards, self.discount_gamma)
        returns = tf.convert_to_tensor(returns, dtype='float32')
        observations = observations[0:-1:]  # The last observation is o_t+1, and it's only needed for gae calculation
        ep_loss, ep_gradnorm = self.training_step_actor(observations, actions, advantage_estimate=returns)
        with self.tensorboard_summary.as_default():
            tf.summary.scalar("Training/Actor loss", ep_loss, step=steps)
            tf.summary.scalar("Training/Actor Grad Norm", ep_gradnorm, step=steps)
            tf.summary.histogram("Training/Returns", returns, step=steps)
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

        gradients = tape.gradient(loss, self.actor_trainable_vars)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor_trainable_vars))
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
    agent.learn(max_timesteps=250000, render_every_n_episode=10)
    agent.test(10)


