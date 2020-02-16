import gym
from agents.PPO import PPO
from utils.env_utils import print_env_info

env = gym.make('LunarLander-v2')
print_env_info(env)
agent = PPO(env)
agent.learn(max_timesteps=200e3, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward: -260.049 | Actor Loss:  -0.0271 | Critic Loss: 1412.3646 | Steps: 102.5 | Total steps:  1025
# ...
# Episode  140 | Reward:  35.517 | Actor Loss:  -0.0077 | Critic Loss: 217.9198 | Steps: 721.3 | Total steps:  45861
# Episode  150 | Reward:  -4.646 | Actor Loss:  -0.0089 | Critic Loss: 235.6432 | Steps: 675.4 | Total steps:  52615
# ...
# Episode  200 | Reward:  67.234 | Actor Loss:  -0.0067 | Critic Loss:  40.1429 | Steps: 975.8 | Total steps:  99553
# Episode  210 | Reward:  56.345 | Actor Loss:  -0.0046 | Critic Loss:  90.3427 | Steps: 842.6 | Total steps:  107979
# ...
# Episode  250 | Reward:  78.594 | Actor Loss:  -0.0063 | Critic Loss:  68.9236 | Steps: 892.8 | Total steps:  145292
# Episode  260 | Reward: 108.686 | Actor Loss:  -0.0061 | Critic Loss:  47.1562 | Steps: 944.3 | Total steps:  154735
# Episode  270 | Reward:  92.106 | Actor Loss:  -0.0071 | Critic Loss:  21.0527 | Steps: 1000.0 | Total steps:  164735
# Episode  280 | Reward:  57.155 | Actor Loss:  -0.0058 | Critic Loss:  87.2131 | Steps: 833.1 | Total steps:  173066
# Episode  290 | Reward:  31.848 | Actor Loss:  -0.0070 | Critic Loss:  84.2972 | Steps: 918.5 | Total steps:  182251
# Episode  300 | Reward:  79.939 | Actor Loss:  -0.0049 | Critic Loss:  25.4500 | Steps: 1000.0 | Total steps:  192251
