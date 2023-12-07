# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module is not found")


class MeanVarBanditEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """environment to test if safe RL algorithms
    prefer a policy that achieves lower variance return"""

    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._size = 2
        self._rng = np.random.RandomState(seed)
        high = np.array([1.0] * self._size, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.idx: Optional[int] = None

    def get_observation(self) -> np.ndarray:
        obs = np.zeros(self._size, dtype=np.float32)
        obs[self.idx] = 1.0
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        super().reset(seed=seed)
        self.idx = 0
        return self.get_observation(), {}

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        if action == 0:
            reward = self.np_random.normal(loc=6.0, scale=1)
        else:
            reward = self.np_random.normal(loc=10.0, scale=3)
        done = True
        observation = self.get_observation()
        return observation, reward, done, False, {"risky_sa": int(action == 1)}
