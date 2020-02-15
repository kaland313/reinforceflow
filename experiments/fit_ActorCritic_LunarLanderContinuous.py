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
# Episode   10 | Reward: -235.399 | Actor Loss:   0.0411 | Critic Loss: 4111.4365 | Steps: 101.8 | Total steps:  1018
# ...
# Episode  260 | Reward: -193.985 | Actor Loss:   0.0130 | Critic Loss: 406.0867 | Steps: 496.2 | Total steps:  51729
# Episode  270 | Reward: -86.613 | Actor Loss:  -0.0459 | Critic Loss: 480.8434 | Steps: 350.8 | Total steps:  55237
# ...
# Episode  330 | Reward: -142.995 | Actor Loss:  -0.0692 | Critic Loss: 369.9376 | Steps: 843.0 | Total steps:  95363
# Episode  340 | Reward: -23.626 | Actor Loss:  -0.0712 | Critic Loss: 204.2929 | Steps: 500.3 | Total steps:  100366
# ...
# Episode  440 | Reward: -27.583 | Actor Loss:  -0.0548 | Critic Loss:  74.0368 | Steps: 938.4 | Total steps:  194615
# Episode  450 | Reward: -39.442 | Actor Loss:  -0.1017 | Critic Loss:  95.4745 | Steps: 1000.0 | Total steps:  204615
# Episode  460 | Reward: -46.753 | Actor Loss:  -0.0627 | Critic Loss: 139.1243 | Steps: 939.5 | Total steps:  214010
# Episode  470 | Reward: -32.150 | Actor Loss:  -0.1046 | Critic Loss: 155.4516 | Steps: 895.0 | Total steps:  222960
# Episode  480 | Reward: -13.466 | Actor Loss:  -0.0965 | Critic Loss:  85.5973 | Steps: 940.3 | Total steps:  232363
# Episode  490 | Reward: -29.066 | Actor Loss:  -0.0602 | Critic Loss:  56.5770 | Steps: 1000.0 | Total steps:  242363

###############################################################
# Notes
# - Action histograms show that only -1 and 1 are sampled
# - Mean is mostly around 0. and is always inside [-1, 1]
# - Std is large (hundreds), and slowly gets unstable around 100-120k steps
# - Tahn transforming the normal distribution solved these problems and converges to ~0 return
#   > The agent is almost capable of softly landing the spacecraft but it doesn't turn off the thrusters upon touchdown
#


