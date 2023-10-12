import logging
from typing import Iterable

try:
    import gymnasium as gym

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym

    logging.warning("Using deprecated 'gym' package.")

import numpy as np
import torch
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation


class GymEnvironment(Environment):
    """
    A wrapper for `gym.Env` (Gym 0.21) to behave like Pearl's `Environment`.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the wrapper to be around a `gym.Env`.
        Args:
            args: positional arguments to be passed on to Gym's environment.
            kwargs: keyword arguments to be passed on to Gym's environment.
        """
        # pyre-fixme[4]: Attribute must be annotated.
        self.env = gym.make(*args, **kwargs)

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def action_space(self):
        return self.env.action_space

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def observation_space(self):
        return self.env.observation_space

    # pyre-fixme[31]: Expression `ActionSpace)` is not a valid type.
    def reset(self) -> (Observation, ActionSpace):
        reset_result = self.env.reset()
        # pyre-fixme[16]: `Iterable` has no attribute `__getitem__`.
        if isinstance(reset_result, Iterable) and isinstance(reset_result[1], dict):
            # newer Gym versions return an info dict.
            observation, info = reset_result
        else:
            observation = reset_result
            # newer Gym versions return an info dict.
            observation = list(observation.values())[0]
        if isinstance(observation, np.ndarray):
            observation = observation.astype(np.float32)
        return observation, self.action_space

    def step(self, action: Action) -> ActionResult:
        # Some versions of Gym do not work with tensor-represented actions,
        # so we ensure it is a numpy array.
        effective_action = (
            action.numpy(force=True) if isinstance(action, torch.Tensor) else action
        )
        reaction = self.env.step(effective_action)
        if len(reaction) == 4:
            # Older Gym versions still using 'done' as opposed to 'terminated' and 'truncated'
            observation, reward, done, info = reaction
            terminated = done
            truncated = False
        elif len(reaction) == 5:
            observation, reward, terminated, truncated, info = reaction
        else:
            raise ValueError(
                f"Unexpected action result from Gym (expected 4 or 5 elements): {reaction}"
            )
        if isinstance(observation, np.ndarray):
            observation = observation.astype(np.float32)
        if isinstance(reward, np.float64):
            reward = reward.astype(np.float32)
        return ActionResult(observation, reward, terminated, truncated, info)

    # pyre-fixme[3]: Return type must be annotated.
    def render(self):
        return self.env.render()

    # pyre-fixme[3]: Return type must be annotated.
    def close(self):
        return self.env.close()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self) -> str:
        return self.env.spec.id
