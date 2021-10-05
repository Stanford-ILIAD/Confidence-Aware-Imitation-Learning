import gym
import numpy as np

gym.logger.set_level(40)


class NormalizedEnv(gym.Wrapper):
    """
    Environment with action space normalized

    Parameters
    ----------
    env: gym.wrappers.TimeLimit
    """

    def __init__(self, env: gym.wrappers.TimeLimit):
        gym.Wrapper.__init__(self, env)
        self.scale = (env.action_space.high - env.action_space.low) / 2.
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action: np.array):
        next_state, reward, done, info = self.env.step(action * self.scale)
        return next_state, reward, done, info

    @property
    def max_episode_steps(self):
        return self.env._max_episode_steps


def make_env(env_id: str) -> NormalizedEnv:
    """
    Make normalized environment

    Parameters
    ----------
    env_id: str
        id of the env

    Returns
    -------
    env: NormalizedEnv
        normalized environment
    """
    return NormalizedEnv(gym.make(env_id))
