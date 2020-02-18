import gym
from agents.PPO import PPO
from utils.env_utils import print_env_info

env = gym.make('MountainCarContinuous-v0')
print_env_info(env)
agent = PPO(env)
agent.learn(max_timesteps=250e3, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results

