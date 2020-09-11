import gym
from agents.PPO import PPO
from utils.env_utils import print_env_info

env = gym.make('LunarLanderContinuous-v2')
print_env_info(env)
agent = PPO(env, rollouts_per_trajectory=5, global_std_for_gaussian_policy=True, tanh_transform_gaussian_policy=False)
agent.learn(max_timesteps=250e3, render_every_n_episode=100000)
agent.save_video()
agent.test(10)

###############################################################
# Results
# Episode   10 | Rollout Reward Sum: -160.132 | Actor Loss:  -0.0169 | Critic Loss: 993.8369 | Steps: 512.8 | Total steps:  5128
# Episode   20 | Rollout Reward Sum: -104.169 | Actor Loss:  -0.0128 | Critic Loss: 524.0778 | Steps: 736.1 | Total steps:  12489
# Episode   30 | Rollout Reward Sum: -82.702 | Actor Loss:  -0.0105 | Critic Loss: 331.7563 | Steps: 1337.8 | Total steps:  25867
# Episode   40 | Rollout Reward Sum: -45.223 | Actor Loss:  -0.0074 | Critic Loss: 164.4782 | Steps: 3196.5 | Total steps:  57832
# Episode   50 | Rollout Reward Sum:  11.476 | Actor Loss:  -0.0047 | Critic Loss:  67.3164 | Steps: 4913.1 | Total steps:  106963
# Episode   60 | Rollout Reward Sum: 106.737 | Actor Loss:  -0.0036 | Critic Loss:  45.7503 | Steps: 4839.4 | Total steps:  155357
# Episode   70 | Rollout Reward Sum: 138.187 | Actor Loss:  -0.0044 | Critic Loss:  90.3441 | Steps: 4206.8 | Total steps:  197425
# Episode   80 | Rollout Reward Sum: 210.782 | Actor Loss:  -0.0054 | Critic Loss: 160.9778 | Steps: 2867.2 | Total steps:  226097
# Episode   90 | Rollout Reward Sum: 222.627 | Actor Loss:  -0.0055 | Critic Loss: 142.7507 | Steps: 2601.7 | Total steps:  252114