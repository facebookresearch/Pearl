# pyre-ignore-all-errors
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module is not found")


class StochMDPEnv(gym.Env):
    """environment to test if safe RL algorithms
    prefer a policy that achieves lower variance return"""

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._size = 2
        self._rng = np.random.RandomState(seed)
        high = np.array([1.0] * self._size, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.reset()

    def get_observation(self):
        obs = np.zeros(self._size, dtype=np.float32)
        obs[self.idx] = 1.0
        return obs

    def reset(self):
        self.idx = 0
        return self.get_observation(), {}

    def step(self, action: int):
        reward = 0.0
        if action == 0:
            reward = np.random.normal(loc=8.0, scale=1)
        else:
            reward = np.random.normal(loc=10.0, scale=3)
        done = True
        observation = self.get_observation()
        return observation, reward, done, False, {}
