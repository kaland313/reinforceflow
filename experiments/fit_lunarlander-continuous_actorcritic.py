import gym
from agents.ActorCritic import ActorCritic
from utils.env_utils import print_env_info

env = gym.make('LunarLanderContinuous-v2')
print_env_info(env)
agent = ActorCritic(env)
agent.learn(max_timesteps=250000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward: -246.791 | Actor Loss:  -0.0070 | Critic Loss: 5120.4507 | Steps: 187.8 | Total steps:  1878
# ...
# Episode  350 | Reward: -153.322 | Actor Loss:  -0.4541 | Critic Loss: 1279.1643 | Steps: 152.9 | Total steps:  50324
# ...
# Episode  650 | Reward: -281.267 | Actor Loss:  -0.4283 | Critic Loss: 1665.2347 | Steps: 169.8 | Total steps:  100337
# ...
# Episode 1050 | Reward: -326.280 | Actor Loss:  -1.4451 | Critic Loss: 2930.1692 | Steps: 112.9 | Total steps:  150558
# ...
# Episode 1500 | Reward: -311.703 | Actor Loss:  -2.3699 | Critic Loss: 1561.8054 | Steps: 114.5 | Total steps:  205292
# Episode 1510 | Reward: -463.753 | Actor Loss:      nan | Critic Loss: 14207.5234 | Steps: 101.3 | Total steps:  206305

###############################################################
# Notes
# - Action histograms show that only -1 and 1 are sampled
# - Mean is mostly around 0. and is always inside [-1, 1]
# - Std is large (hundreds), and slowly gets unstable around 100-120k steps

