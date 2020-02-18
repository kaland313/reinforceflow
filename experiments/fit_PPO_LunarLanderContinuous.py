import gym
from agents.PPO import PPO
from utils.env_utils import print_env_info

env = gym.make('LunarLanderContinuous-v2')
print_env_info(env)
agent = PPO(env, clip_epsilon=0.2)
agent.learn(max_timesteps=250e3, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward: -511.040 | Actor Loss:  -0.0392 | Critic Loss: 2690.0754 | Steps: 93.2 | Total steps:   932
# Episode   20 | Reward: -1166.660 | Actor Loss:   0.0000 | Critic Loss: 5519.8096 | Steps: 88.2 | Total steps:  1814
# Episode   30 | Reward: -968.318 | Actor Loss:  -0.0000 | Critic Loss: 4557.6084 | Steps: 78.3 | Total steps:  2597
# Episode   40 | Reward: -1100.056 | Actor Loss:  -0.0000 | Critic Loss: 6053.2480 | Steps: 86.2 | Total steps:  3459
# Episode   50 | Reward: -877.862 | Actor Loss:  -0.0000 | Critic Loss: 2795.5522 | Steps: 70.1 | Total steps:  4160
# Episode   60 | Reward: -1158.415 | Actor Loss:   0.0000 | Critic Loss: 3428.4321 | Steps: 90.7 | Total steps:  5067

