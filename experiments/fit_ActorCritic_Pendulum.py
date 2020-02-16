import gym
from agents.ActorCritic import ActorCritic
from utils.env_utils import print_env_info

env = gym.make('Pendulum-v0')
print_env_info(env)
agent = ActorCritic(env)
agent.learn(max_timesteps=75000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward: -1417.347 | Actor Loss:      nan | Critic Loss: 27245.2441 | Steps: 200.0 | Total steps:  2000


