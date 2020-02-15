import gym
from agents.ActorCritic import ActorCritic
from utils.env_utils import print_env_info

env = gym.make('LunarLander-v2')
print_env_info(env)
agent = ActorCritic(env)
agent.learn(max_timesteps=250000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Reward: -197.282 | Actor Loss:   0.0055 | Critic Loss: 3946.6909 | Steps: 79.5 | Total steps:   795
# ...
# Episode  350 | Reward: -91.329 | Actor Loss:  -0.0490 | Critic Loss: 439.6674 | Steps: 394.6 | Total steps:  49870
# Episode  360 | Reward: -128.354 | Actor Loss:  -0.0410 | Critic Loss: 327.3320 | Steps: 265.3 | Total steps:  52523
# ...
# Episode  460 | Reward: -18.757 | Actor Loss:  -0.0495 | Critic Loss: 177.4417 | Steps: 799.4 | Total steps:  97099
# Episode  470 | Reward: -10.284 | Actor Loss:  -0.0731 | Critic Loss: 267.7109 | Steps: 679.9 | Total steps:  103898
# ...
# Episode  540 | Reward:   4.791 | Actor Loss:  -0.0636 | Critic Loss: 229.3001 | Steps: 635.4 | Total steps:  146131
# Episode  550 | Reward:  33.934 | Actor Loss:  -0.0575 | Critic Loss: 164.3399 | Steps: 869.0 | Total steps:  154821
# ...
# Episode  610 | Reward:  48.781 | Actor Loss:  -0.0741 | Critic Loss: 186.2074 | Steps: 753.2 | Total steps:  203949
# ...
# Episode  630 | Reward:  55.447 | Actor Loss:  -0.0560 | Critic Loss:  99.3020 | Steps: 941.9 | Total steps:  221822
# Episode  640 | Reward:  71.505 | Actor Loss:  -0.0606 | Critic Loss:  70.3705 | Steps: 976.5 | Total steps:  231587
# Episode  650 | Reward:  45.553 | Actor Loss:  -0.0672 | Critic Loss:  92.7040 | Steps: 887.7 | Total steps:  240464
# Episode  660 | Reward: -14.033 | Actor Loss:  -0.0620 | Critic Loss:  81.8725 | Steps: 954.0 | Total steps:  250004


