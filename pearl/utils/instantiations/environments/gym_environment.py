# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.space import Space
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import Tensor
import torch

try:
    import gymnasium as gym

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym

    logging.warning("Using deprecated 'gym' package.")


def single_element_tensor_to_int(x: Tensor) -> int:
    return int(x)


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy(force=True)


GYM_TO_PEARL_ACTION_SPACE = {
    "Discrete": DiscreteActionSpace,
    "Box": BoxActionSpace,
    # Add more here as needed
}
GYM_TO_PEARL_OBSERVATION_SPACE = {
    "Discrete": DiscreteSpace,
    "Box": BoxSpace,
    # Add more here as needed
}
# pyre-fixme[5]: Global expression must be annotated.
PEARL_TO_GYM_ACTION = {
    "Discrete": single_element_tensor_to_int,
    "Box": tensor_to_numpy,
    # Add more here as needed
}


class GymEnvironment(Environment):
    """A wrapper for `gym.Env` to behave like Pearl's `Environment`."""

    def __init__(
        self, env_or_env_name: gym.Env | str, *args: Any, **kwargs: Any
    ) -> None:
        """Constructs a `GymEnvironment` wrapper.

        Args:
            env_or_env_name: A gym.Env instance or a name of a gym.Env.
            args: Arguments passed to `gym.make()` if the first argument is a string.
            kwargs: Keyword arguments passed to `gym.make()` if the first argument is a string.
        """
        if type(env_or_env_name) is str:
            env = gym.make(env_or_env_name, *args, **kwargs)
        else:
            env = env_or_env_name
        self.env: gym.Env = env
        self._action_space: ActionSpace = _get_pearl_space(
            gym_space=self.env.action_space,
            gym_to_pearl_map=GYM_TO_PEARL_ACTION_SPACE,
        )
        self._observation_space: Space = _get_pearl_space(
            gym_space=self.env.observation_space,
            gym_to_pearl_map=GYM_TO_PEARL_OBSERVATION_SPACE,
        )

    @property
    def action_space(self) -> ActionSpace:
        """Returns the Pearl action space for this environment."""
        return self._action_space

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def reset(self, seed: int | None = None) -> tuple[Observation, ActionSpace]:
        """Resets the environment and returns the initial observation and
        initial action space."""

        # pyre-fixme[16]: `ActionSpace` has no attribute `gym_space`.
        self._action_space.gym_space.seed(seed)
        self.env.action_space.seed(seed)
        reset_result = self.env.reset()
        if isinstance(reset_result, Iterable) and isinstance(reset_result[1], dict):
            # newer Gym versions return an info dict.
            observation, info = self.env.reset(seed=seed)
        else:
            # TODO: Deprecate this part at some point and only support new
            # version of Gymnasium?
            observation = list(reset_result.values())[0]  # pyre-ignore
        if isinstance(observation, np.ndarray) and observation.dtype == np.float64:
            observation = observation.astype(np.float32)
        return observation, self.action_space

    def step(self, action: Action) -> ActionResult:
        """Takes one step in the environment given the agent's action. Returns an
        `ActionResult` object containing the next observation, reward, and done flag."""
        # Convert action to the format expected by Gymnasium
        effective_action = _get_gym_action(
            pearl_action=action, gym_space=self.env.action_space
        )
        # Take a step in the environment and receive an action result
        gym_action_result = self.env.step(effective_action)
        if len(gym_action_result) == 4:
            # Older Gym versions use 'done' as opposed to 'terminated' and 'truncated'
            observation, reward, done, info = gym_action_result  # pyre-ignore
            if done:
                truncated = info["TimeLimit.truncated"]
                terminated = not truncated
            else:
                truncated = False
                terminated = False
        elif len(gym_action_result) == 5:
            # Newer Gym versions use 'terminated' and 'truncated'
            observation, reward, terminated, truncated, info = gym_action_result
        else:
            raise ValueError(
                f"Unexpected action result from Gym (expected 4 or 5 elements): {gym_action_result}"
            )
        if "cost" in info.keys():
            cost = info["cost"]
        else:
            cost = None

        if "available_action_space" in info.keys():
            available_action_space = info["available_action_space"]
        else:
            available_action_space = None

        if isinstance(observation, np.ndarray) and observation.dtype == np.float64:
            observation = observation.astype(np.float32)
        if isinstance(reward, np.float64):
            reward = reward.astype(np.float32)
        if isinstance(cost, np.float64):
            cost = cost.astype(np.float32)

        return ActionResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            cost=cost,
            available_action_space=available_action_space,
        )

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def __str__(self) -> str:
        if self.env.spec is not None:
            return self.env.spec.id
        else:
            return "CustomGymEnvironment"


def _get_gym_action(
    pearl_action: Action,
    gym_space: gym.Space,
    # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
) -> int | np.ndarray:
    """A helper function to convert a Pearl `Action` to an action compatible with
    the Gym action space `gym_space`."""
    gym_space_name = gym_space.__class__.__name__
    try:
        pearl_to_gym_action_transform = PEARL_TO_GYM_ACTION[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return pearl_to_gym_action_transform(pearl_action)


def _tuple_to_box_space(gym_space: gym.Space) -> BoxSpace:
    """Converts a Gymnasium ``Tuple`` space into a ``BoxSpace`` by concatenating
    the bounds of each subspace."""
    assert gym_space.__class__.__name__ == "Tuple"
    lows = []
    highs = []
    for subspace in gym_space.spaces:  # pyre-ignore[16]
        name = subspace.__class__.__name__
        if name == "Box":
            lows.append(torch.tensor(subspace.low).flatten())
            highs.append(torch.tensor(subspace.high).flatten())
        elif name == "Discrete":
            start = getattr(subspace, "start", 0)
            lows.append(torch.tensor([float(start)]))
            highs.append(torch.tensor([float(start + subspace.n - 1)]))
        else:
            raise NotImplementedError(f"Unsupported subspace type: {name}")
    low = torch.cat(lows).float()
    high = torch.cat(highs).float()
    return BoxSpace(low=low, high=high)


def _get_pearl_space(
    gym_space: gym.Space, gym_to_pearl_map: dict[str, Any]
) -> ActionSpace:
    """Returns the Pearl action space for this environment."""
    gym_space_name = gym_space.__class__.__name__
    if gym_space_name == "Tuple":
        return _tuple_to_box_space(gym_space)
    try:
        pearl_action_space_cls = gym_to_pearl_map[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return pearl_action_space_cls.from_gym(gym_space)
