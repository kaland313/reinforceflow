

from agents.ActorCritic import ActorCritic


class PPO(ActorCritic):
    def __init__(self,
                 env,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-2,
                 discount_gamma=0.99,
                 generalized_advantage_estimate_lambda=0.97):
        super(PPO, self).__init__(env, actor_learning_rate, critic_learning_rate, discount_gamma,
                                  generalized_advantage_estimate_lambda)



