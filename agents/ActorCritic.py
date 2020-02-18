import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
from agents.PolicyGradient import PolicyGradient
from utils.reward_calc import calculate_generalized_advantage_estimate, calculate_discounted_returns


class ActorCritic(PolicyGradient):
    def __init__(self,
                 env,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-2,
                 discount_gamma=0.99,
                 generalized_advantage_estimate_lambda=0.97,
                 rollouts_per_trajectory=1,
                 global_std_for_gaussian_policy=False,
                 tanh_transform_gaussian_policy=True,
                 algo_str="A2C"):
        super(ActorCritic, self).__init__(env, actor_learning_rate, discount_gamma, rollouts_per_trajectory,
                                          global_std_for_gaussian_policy, tanh_transform_gaussian_policy,
                                          algo_str=algo_str)
        self.critic_learning_rate = critic_learning_rate
        self.gae_lambda = generalized_advantage_estimate_lambda

        self.critic_model = None  # type: tf.keras.Model
        self.setup_critic_model()
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.critic_learning_rate)
        self.critic_loss = tf.keras.losses.MeanSquaredError()

    def setup_critic_model(self):
        self.critic_model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.env.observation_space.shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])  # kernel_initializer=tf.keras.initializers.Zeros

    def training_step(self, observations, actions, rewards, dones, steps):
        values = tf.squeeze(self.critic_model(observations))
        advantages, returns = calculate_generalized_advantage_estimate(rewards, values, dones,
                                                                       self.gae_lambda, self.discount_gamma)
        # values = tf.squeeze(self.critic_model(observations))
        # returns = calculate_discounted_returns(rewards,self.discount_gamma)
        # advantages = returns - values

        actor_loss, actor_gradnorm = self.training_step_actor(observations, actions, advantages)
        critic_loss, critic_gradnorm = self.training_step_critic(observations, returns)

        with self.tensorboard_summary.as_default():
            tf.summary.scalar("Training/Actor loss", actor_loss, step=steps)
            tf.summary.scalar("Training/Actor Grad Norm", actor_gradnorm, step=steps)
            tf.summary.scalar("Training/Critic loss", critic_loss, step=steps)
            tf.summary.scalar("Training/Critic Grad Norm", critic_gradnorm, step=steps)
            tf.summary.histogram("Training/Advantages", advantages, step=steps)
            tf.summary.histogram("Training/Returns", returns, step=steps)
        return actor_loss, critic_loss

    @tf.function(experimental_relax_shapes=True)
    def training_step_critic(self, observations, targets):
        with tf.GradientTape() as tape:
            predictions = self.critic_model(observations)
            loss = self.critic_loss(targets, predictions)
        gradients = tape.gradient(loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic_model.trainable_variables))
        return tf.reduce_mean(loss), tf.linalg.global_norm(gradients)
