import gym
from reinforceflow.agents.actor_critic import ActorCritic
from reinforceflow.utils.env_utils import print_env_info

env = gym.make('LunarLanderContinuous-v2')
print_env_info(env)
agent = ActorCritic(env, global_std_for_gaussian_policy=True, tanh_transform_gaussian_policy=False)
agent.learn(max_timesteps=250000, render_every_n_episode=100000)
agent.test(10)

###############################################################
# Results
# Episode   10 | Rollout Reward Sum: -222.853 | Actor Loss:  -0.0089 | Critic Loss: 4078.2476 | Steps: 105.8 | Total steps:  1058
# ...
# Episode  300 | Rollout Reward Sum: -120.277 | Actor Loss:  -0.0385 | Critic Loss: 671.9962 | Steps: 274.6 | Total steps:  48092
# Episode  310 | Rollout Reward Sum: -88.077 | Actor Loss:  -0.0230 | Critic Loss: 569.6835 | Steps: 355.9 | Total steps:  51651
# ...
# Episode  410 | Rollout Reward Sum: -20.237 | Actor Loss:  -0.0021 | Critic Loss: 197.7371 | Steps: 389.6 | Total steps:  99121
# Episode  420 | Rollout Reward Sum: -172.291 | Actor Loss:   0.0073 | Critic Loss: 296.2503 | Steps: 629.9 | Total steps:  105420
# ...
# Episode  470 | Rollout Reward Sum: -14.090 | Actor Loss:  -0.0080 | Critic Loss: 128.1878 | Steps: 918.4 | Total steps:  145725
# Episode  480 | Rollout Reward Sum: -75.831 | Actor Loss:  -0.0047 | Critic Loss: 105.8857 | Steps: 900.6 | Total steps:  154731
# ...
# Episode  540 | Rollout Reward Sum:  11.979 | Actor Loss:  -0.0221 | Critic Loss: 337.3446 | Steps: 441.0 | Total steps:  205384
# Episode  550 | Rollout Reward Sum:  55.535 | Actor Loss:  -0.0016 | Critic Loss: 169.7236 | Steps: 753.5 | Total steps:  212919
# Episode  560 | Rollout Reward Sum:  -0.770 | Actor Loss:  -0.0073 | Critic Loss: 162.3494 | Steps: 710.9 | Total steps:  220028
# Episode  570 | Rollout Reward Sum:  44.120 | Actor Loss:   0.0179 | Critic Loss:  99.7854 | Steps: 933.9 | Total steps:  229367
# Episode  580 | Rollout Reward Sum:  40.557 | Actor Loss:  -0.0018 | Critic Loss: 123.3792 | Steps: 905.5 | Total steps:  238422
# Episode  590 | Rollout Reward Sum:  17.012 | Actor Loss:  -0.0189 | Critic Loss: 117.6929 | Steps: 960.1 | Total steps:  248023