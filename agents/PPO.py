import tensorflow as tf
from utils.reward_calc import calculate_generalized_advantage_estimate, safe_normalize_tf

from agents.ActorCritic import ActorCritic


class PPO(ActorCritic):
    def __init__(self,
                 env,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-2,
                 discount_gamma=0.99,
                 generalized_advantage_estimate_lambda=0.97,
                 global_std_for_gaussian_policy=False,
                 tanh_transform_gaussian_policy=True,
                 algo_str="PPO",
                 clip_epsilon=0.2,
                 policy_train_epochs=10):
        super(PPO, self).__init__(env, actor_learning_rate, critic_learning_rate, discount_gamma,
                                  generalized_advantage_estimate_lambda, global_std_for_gaussian_policy,
                                  tanh_transform_gaussian_policy, algo_str)
        self.clip_epsilon = clip_epsilon
        self.policy_train_epochs = policy_train_epochs

    def training_step(self, observations, actions, rewards, dones, steps):
        values = tf.squeeze(self.critic_model(observations))
        advantages, returns = calculate_generalized_advantage_estimate(rewards, values, dones,
                                                                       self.gae_lambda, self.discount_gamma)
        observations = observations[0:-1]  # The last observation is o_t+1, and it's only needed for gae calculation

        old_network_output = self.actor_model(observations)
        old_neg_log_prob_a_t = self.proba_distribution.neg_log_prob_a_t(old_network_output, actions)
        for epochs in range(self.policy_train_epochs):
            actor_loss, actor_gradnorm = self.training_step_actor(observations, actions, advantages,
                                                                  old_neg_log_prob_a_t)
            critic_loss, critic_gradnorm = self.training_step_critic(observations, returns)

        with self.tensorboard_summary.as_default():
            tf.summary.scalar("Training/Actor loss", actor_loss, step=steps)
            tf.summary.scalar("Training/Actor Grad Norm", actor_gradnorm, step=steps)
            tf.summary.scalar("Training/Critic loss", critic_loss, step=steps)
            tf.summary.scalar("Training/Critic Grad Norm", critic_gradnorm, step=steps)
            tf.summary.histogram("Training/Advantages", advantages, step=steps)
            tf.summary.histogram("Training/Returns", returns, step=steps)
        return actor_loss, critic_loss


    # @tf.function(experimental_relax_shapes=True)
    def training_step_actor(self, observations, actions, advantage_estimate, old_neg_log_prob_a_t=None):
        assert old_neg_log_prob_a_t is not None
        normalized_advantages = safe_normalize_tf(advantage_estimate)

        # According to https://arxiv.org/abs/1707.06347: the probability ratioris clipped at 1−epsion or 1 + epsilon
        # depending on whether the advantage is positive or negative"
        # pos_advantage = tf.cast(normalized_advantages > 0, dtype='float32')
        # active_clip_limit = (1. + self.clip_epsilon) * pos_advantage + (1. - self.clip_epsilon) * (
        #         1. - pos_advantage)
        # clipped_term = active_clip_limit * advantage_estimate

        with tf.GradientTape() as tape:
            network_output = self.actor_model(observations)
            neg_log_prob_at = self.proba_distribution.neg_log_prob_a_t(network_output, actions)
            # Both neg_log_prob_a_ts must be multiplied by -1 because prob_ratio = exp(log_prob_a_t - old_log_prob_a_t)
            prob_ratio = tf.exp(-neg_log_prob_at + old_neg_log_prob_a_t)
            L_CPI = prob_ratio * normalized_advantages
            clipped_term = tf.clip_by_value(prob_ratio, 1. - self.clip_epsilon, 1. + self.clip_epsilon) * normalized_advantages
            L_CLIP = tf.reduce_mean(tf.math.minimum(L_CPI, clipped_term))
            loss = -L_CLIP

        gradients = tape.gradient(loss, self.actor_trainable_vars)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor_trainable_vars))
        return tf.reduce_mean(loss), tf.linalg.global_norm(gradients)


