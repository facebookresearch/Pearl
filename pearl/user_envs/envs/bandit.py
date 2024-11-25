# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module is not found")


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
class MeanVarBanditEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """environment to test if safe RL algorithms
    prefer a policy that achieves lower variance return"""

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._size = 2
        self._rng = np.random.RandomState(seed)
        high = np.array([1.0] * self._size, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.idx: int | None = None

    # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    def get_observation(self) -> np.ndarray:
        obs = np.zeros(self._size, dtype=np.float32)
        obs[self.idx] = 1.0
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, float] | None = None,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        self.idx = 0
        return self.get_observation(), {}

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def step(
        self,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        action: int | np.ndarray,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 0.0
        if action == 0:
            reward = self.np_random.normal(loc=6.0, scale=1)
        else:
            reward = self.np_random.normal(loc=10.0, scale=3)
        terminated = True
        observation = self.get_observation()
        return observation, reward, terminated, False, {"risky_sa": int(action == 1)}
