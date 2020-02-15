import gym


def print_env_info(env: gym.Env):
    print(env)
    if isinstance(env.action_space, gym.spaces.Box):
        print("Action space: ", env.action_space, " [", env.action_space.low, ", ", env.action_space.high, "]")
    else:
        print("Action space: ", env.action_space)
    if isinstance(env.observation_space, gym.spaces.Box):
        print("Observation space:", env.observation_space,
              " [", env.observation_space.low, ", ", env.observation_space.high, "]")
    else:
        print("Observation space:", env.observation_space)