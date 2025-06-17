import unittest
import numpy as np
import torch

from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.instantiations.environments.environments import FlattenDictObservations

try:
    import gymnasium as gym
except ModuleNotFoundError:  # fallback for gym
    import gym


class DummyDictEnv(Environment):
    def __init__(self) -> None:
        self._action_space = DiscreteActionSpace([torch.tensor(0), torch.tensor(1)])
        self._observation_space = gym.spaces.Dict(
            {
                "x": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "y": gym.spaces.Discrete(3),
            }
        )

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    def reset(self, seed: int | None = None) -> tuple[Observation, ActionSpace]:
        obs = {"x": np.zeros(2, dtype=np.float32), "y": 1}
        return obs, self.action_space

    def step(self, action: Action) -> ActionResult:
        obs = {"x": np.zeros(2, dtype=np.float32), "y": 1}
        return ActionResult(observation=obs, reward=0.0, terminated=True, truncated=False, info={})


class TestFlattenDictObservations(unittest.TestCase):
    def test_flatten(self) -> None:
        env = FlattenDictObservations(DummyDictEnv())
        obs, _ = env.reset()
        self.assertIsInstance(obs, torch.Tensor)
        # 2 from "x" and 1 from "y"
        self.assertEqual(obs.shape[0], 3)
        step_result = env.step(torch.tensor(0))
        self.assertTrue(torch.equal(step_result.observation, obs))


if __name__ == "__main__":
    unittest.main()