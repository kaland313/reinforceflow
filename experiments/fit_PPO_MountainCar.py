import gym
from reinforceflow.agents.ppo import PPO
from reinforceflow.utils.env_utils import print_env_info

env = gym.make('MountainCar-v0')
print_env_info(env)
agent = PPO(env, global_std_for_gaussian_policy=True, tanh_transform_gaussian_policy=False)
agent.learn(max_timesteps=250e3, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results