import gym
from agents.ActorCritic import ActorCritic
from utils.env_utils import print_env_info

env = gym.make('CartPole-v1')
print_env_info(env)
agent = ActorCritic(env)
agent.learn(max_timesteps=75000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward:  20.800 | Actor Loss:   0.0049 | Critic Loss:  80.9111 | Steps: 20.8 | Total steps:   208
# ...
# Episode  300 | Reward: 464.400 | Actor Loss:  -0.0207 | Critic Loss: 146.1699 | Steps: 464.4 | Total steps:  44777
# Episode  310 | Reward: 500.000 | Actor Loss:  -0.0130 | Critic Loss: 183.4195 | Steps: 500.0 | Total steps:  49777
# Episode  320 | Reward: 500.000 | Actor Loss:  -0.0083 | Critic Loss: 153.6212 | Steps: 500.0 | Total steps:  54777
# Episode  330 | Reward: 500.000 | Actor Loss:  -0.0131 | Critic Loss: 149.7085 | Steps: 500.0 | Total steps:  59777
# Episode  340 | Reward: 500.000 | Actor Loss:  -0.0096 | Critic Loss: 176.7943 | Steps: 500.0 | Total steps:  64777
# Episode  350 | Reward: 461.600 | Actor Loss:  -0.0063 | Critic Loss: 159.4455 | Steps: 461.6 | Total steps:  69393
# Episode  360 | Reward: 443.900 | Actor Loss:  -0.0075 | Critic Loss: 169.3608 | Steps: 443.9 | Total steps:  73832

