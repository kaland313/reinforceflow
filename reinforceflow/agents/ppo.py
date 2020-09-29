import tensorflow as tf
from ..utils.reward_calc import calculate_generalized_advantage_estimate, safe_normalize_tf

from .actor_critic import ActorCritic


class PPO(ActorCritic):
    def __init__(self,
                 env,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-2,
                 discount_gamma=0.99,
                 generalized_advantage_estimate_lambda=0.97,
                 rollouts_per_trajectory=1,
                 global_std_for_gaussian_policy=False,
                 tanh_transform_gaussian_policy=True,
                 algo_str="PPO",
                 clip_epsilon=0.2,
                 policy_train_epochs=10,
                 kl_target=0.01,
                 init_kl_coef=0.2,
                 use_kl_loss=True):
        super(PPO, self).__init__(env, actor_learning_rate, critic_learning_rate, discount_gamma,
                                  generalized_advantage_estimate_lambda, rollouts_per_trajectory,
                                  global_std_for_gaussian_policy, tanh_transform_gaussian_policy, algo_str)
        self.clip_epsilon = clip_epsilon
        self.policy_train_epochs = policy_train_epochs
        self.kl_target = kl_target              # denoted by d_targ in the PPO paper (default value is from rllib)
        self.cur_kl_coeff = init_kl_coef        # denoted by beta in the PPO paper (default value is from rllib)
        self.use_kl_loss = use_kl_loss

    def training_step(self, observations, actions, rewards, dones, steps):
        values = tf.squeeze(self.critic_model(observations))
        advantages, returns = calculate_generalized_advantage_estimate(rewards, values, dones,
                                                                       self.gae_lambda, self.discount_gamma)

        old_network_output = self.actor_model(observations)
        for epochs in range(self.policy_train_epochs):
            metrics_actor = self.training_step_actor(observations, actions, advantages,
                                                                  old_network_output)
            metrics_critic = self.training_step_critic(observations, returns)
        metrics = {**metrics_actor, **metrics_critic}
        histograms = {"Episode metrics/Returns": returns,
                      "Episode metrics/Advantages": advantages}
        self.log_metrics(metrics, histograms, steps)
        return metrics["Losses/Actor Loss Total"], metrics["Losses/Critic Loss Total"]


    # @tf.function(experimental_relax_shapes=True)
    def training_step_actor(self, observations, actions, advantage_estimate, old_network_output=None):
        assert old_network_output is not None
        old_neg_log_prob_a_t = self.proba_distribution.neg_log_prob_a_t(old_network_output, actions)
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
            L_CLIP = tf.math.minimum(L_CPI, clipped_term)
            if self.use_kl_loss:
                sampled_kl = self.proba_distribution.kl(old_network_output, network_output)
                L_KLPEN = L_CPI - self.cur_kl_coeff * sampled_kl
            else:
                L_KLPEN = 0

            # Combined Clipping and KL penalty loss function as in rllib's implementation
            # https://github.com/ray-project/ray/blob/79c6a6fa02e58aa9432eb2463d093221903cfead/rllib/agents/ppo/ppo_tf_policy.py#L93
            loss = -tf.reduce_mean(L_CLIP + L_KLPEN)

        gradients = tape.gradient(loss, self.actor_trainable_vars)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor_trainable_vars))
        if self.use_kl_loss:
            self.update_kl_coeff(sampled_kl)
        return {"Losses/Actor Loss Total": tf.reduce_mean(loss),
                "Losses/Actor Loss CLIP": -tf.reduce_mean(L_CLIP),
                "Losses/Actor Loss KLPEN": -tf.reduce_mean(L_KLPEN),
                "Losses/Actor Grad Norm": tf.linalg.global_norm(gradients)}


    def update_kl_coeff(self, sampled_kl):
        expected_sampled_kl = tf.reduce_mean(sampled_kl)       # Expected value over a trajectory or over time
        if expected_sampled_kl < self.kl_target / 1.5:
            self.cur_kl_coeff /= 2
        if expected_sampled_kl > self.kl_target * 1.5:
            self.cur_kl_coeff *= 2