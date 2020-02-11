import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
from PolicyGradient import calculate_discounted_returns, PolicyGradient

class ActorCritic(PolicyGradient):
    def __init__(self, env, episode_max_timesteps=300, actor_learning_rate=1e-3, critic_learning_rate=1e-2):
        super(ActorCritic, self).__init__(env, episode_max_timesteps, actor_learning_rate)
        self.critic_learning_rate = critic_learning_rate
        self.critic_model = None  # type: tf.keras.Model3
        self.setup_critic_model()
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.critic_learning_rate)
        self.critic_loss = tf.keras.losses.MeanSquaredError()

    def setup_critic_model(self):
        self.critic_model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.env.observation_space.shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])  # kernel_initializer=tf.keras.initializers.Zeros

    def training_step(self, observations, actions, rewards, returns, steps):
        values = self.critic_model(observations)
        advantages = returns-values
        actor_loss, actor_gradnorm = self.training_step_actor(observations, actions, advantages)
        critic_loss, critic_gradnorm = self.training_step_critic(observations, returns)

        with self.tensorboard_summary.as_default():
            tf.summary.scalar("Training/Actor loss", actor_loss, step=steps)
            tf.summary.scalar("Training/Actor Grad Norm", actor_gradnorm, step=steps)
            tf.summary.scalar("Training/Critic loss", critic_loss, step=steps)
            tf.summary.scalar("Training/Critic Grad Norm", critic_gradnorm, step=steps)
            tf.summary.histogram("Training/Advantages", advantages, step=steps)

        return actor_loss, critic_loss

    @tf.function(experimental_relax_shapes=True)
    def training_step_critic(self, observations, targets):
        with tf.GradientTape() as tape:
            predictions = self.critic_model(observations)
            loss = self.critic_loss(targets, predictions)
        gradients = tape.gradient(loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic_model.trainable_variables))
        return tf.reduce_mean(loss), tf.linalg.global_norm(gradients)

if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('LunarLander-v2')
    env = gym.make('LunarLanderContinuous-v2')
    print(env)
    print("Action space: ", env.action_space, "\nObservation space:", env.observation_space)
    agent = ActorCritic(env, critic_learning_rate=1e-3)
    agent.learn(max_timesteps=250000, render_every_n_episode=50)
    agent.test(10)