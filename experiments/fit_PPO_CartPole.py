import gym
from agents.PPO import PPO
from utils.env_utils import print_env_info

env = gym.make('CartPole-v1')
print_env_info(env)
agent = PPO(env)
agent.learn(max_timesteps=50000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward:  20.000 | Actor Loss:  -0.0191 | Critic Loss:  23.6561 | Steps: 20.0 | Total steps:   200
# Episode   20 | Reward:  21.800 | Actor Loss:  -0.0302 | Critic Loss:  15.4321 | Steps: 21.8 | Total steps:   418
# Episode   30 | Reward:  70.600 | Actor Loss:  -0.0154 | Critic Loss:  41.0728 | Steps: 70.6 | Total steps:  1124
# Episode   40 | Reward: 134.700 | Actor Loss:  -0.0126 | Critic Loss:  69.0876 | Steps: 134.7 | Total steps:  2471
# Episode   50 | Reward: 244.500 | Actor Loss:  -0.0104 | Critic Loss:  76.3024 | Steps: 244.5 | Total steps:  4916
# Episode   60 | Reward: 428.100 | Actor Loss:  -0.0058 | Critic Loss:  49.7348 | Steps: 428.1 | Total steps:  9197
# Episode   70 | Reward: 444.000 | Actor Loss:  -0.0068 | Critic Loss:  51.8385 | Steps: 444.0 | Total steps:  13637
# Episode   80 | Reward: 445.700 | Actor Loss:  -0.0085 | Critic Loss:  12.5242 | Steps: 445.7 | Total steps:  18094
# Episode   90 | Reward: 451.100 | Actor Loss:  -0.0065 | Critic Loss:  33.7261 | Steps: 451.1 | Total steps:  22605
# Episode  100 | Reward: 388.000 | Actor Loss:  -0.0095 | Critic Loss:   4.7357 | Steps: 388.0 | Total steps:  26485
# Episode  110 | Reward: 423.100 | Actor Loss:  -0.0066 | Critic Loss:  72.6156 | Steps: 423.1 | Total steps:  30716
# Episode  120 | Reward: 500.000 | Actor Loss:  -0.0045 | Critic Loss: 149.0522 | Steps: 500.0 | Total steps:  35716
# Episode  130 | Reward: 500.000 | Actor Loss:  -0.0029 | Critic Loss: 133.2729 | Steps: 500.0 | Total steps:  40716
# Episode  140 | Reward: 495.400 | Actor Loss:  -0.0045 | Critic Loss: 131.8945 | Steps: 495.4 | Total steps:  45670
