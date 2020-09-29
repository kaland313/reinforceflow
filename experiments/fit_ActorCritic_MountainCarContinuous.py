import gym
from reinforceflow.agents.actor_critic import ActorCritic
from reinforceflow.utils.env_utils import print_env_info

env = gym.make('MountainCarContinuous-v0')
print_env_info(env)
agent = ActorCritic(env)
agent.learn(max_timesteps=250000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward: -35.269 | Actor Loss:   0.0119 | Critic Loss:   0.5310 | Steps: 999.0 | Total steps:  9990
# Episode   20 | Reward: -28.243 | Actor Loss:  -0.0171 | Critic Loss:   0.3007 | Steps: 999.0 | Total steps:  19980
# Episode   30 | Reward: -21.851 | Actor Loss:  -0.0518 | Critic Loss:   0.0893 | Steps: 999.0 | Total steps:  29970
# Episode   40 | Reward: -14.858 | Actor Loss:  -0.0728 | Critic Loss:   0.0491 | Steps: 999.0 | Total steps:  39960
# Episode   50 | Reward:  -8.896 | Actor Loss:  -0.0967 | Critic Loss:   0.0177 | Steps: 999.0 | Total steps:  49950
# Episode   60 | Reward:  -4.049 | Actor Loss:  -0.0757 | Critic Loss:   0.0101 | Steps: 999.0 | Total steps:  59940
# Episode   70 | Reward:  -1.826 | Actor Loss:  -0.0788 | Critic Loss:   0.0052 | Steps: 999.0 | Total steps:  69930
# Episode   80 | Reward:  -0.824 | Actor Loss:  -0.0623 | Critic Loss:   0.0021 | Steps: 999.0 | Total steps:  79920
# Episode   90 | Reward:  -0.344 | Actor Loss:  -0.0248 | Critic Loss:   0.0007 | Steps: 999.0 | Total steps:  89910
# Episode  100 | Reward:  -0.252 | Actor Loss:  -0.0400 | Critic Loss:   0.0002 | Steps: 999.0 | Total steps:  99900
# Episode  110 | Reward:  -0.103 | Actor Loss:  -0.0312 | Critic Loss:   0.0001 | Steps: 999.0 | Total steps:  109890
# Episode  120 | Reward:  -0.066 | Actor Loss:  -0.0578 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  119880
# Episode  130 | Reward:  -0.134 | Actor Loss:  -0.0363 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  129870
# Episode  140 | Reward:  -0.459 | Actor Loss:   0.0012 | Critic Loss:   0.0001 | Steps: 999.0 | Total steps:  139860
# Episode  150 | Reward:  -0.134 | Actor Loss:  -0.0149 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  149850
# Episode  160 | Reward:  -0.153 | Actor Loss:  -0.0024 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  159840
# Episode  170 | Reward:  -0.196 | Actor Loss:  -0.0418 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  169830
# Episode  180 | Reward:  -0.201 | Actor Loss:  -0.0147 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  179820
# Episode  190 | Reward:  -0.266 | Actor Loss:  -0.0191 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  189810
# Episode  200 | Reward:  -0.441 | Actor Loss:  -0.0052 | Critic Loss:   0.0001 | Steps: 999.0 | Total steps:  199800
# Episode  210 | Reward:  -0.204 | Actor Loss:  -0.0426 | Critic Loss:   0.0000 | Steps: 999.0 | Total steps:  209790