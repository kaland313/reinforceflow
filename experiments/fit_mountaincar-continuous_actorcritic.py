import gym
from agents.ActorCritic import ActorCritic
from utils.env_utils import print_env_info

env = gym.make('MountainCarContinuous-v0-v2')
print_env_info(env)
agent = ActorCritic(env)
agent.learn(max_timesteps=250000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
