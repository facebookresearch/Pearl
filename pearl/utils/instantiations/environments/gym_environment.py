import logging
from typing import Any, Iterable, Tuple, Union

import numpy as np
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.utils.instantiations.action_spaces.box import BoxActionSpace
from pearl.utils.instantiations.action_spaces.discrete import DiscreteActionSpace
from torch import Tensor

try:
    import gymnasium as gym

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym

    logging.warning("Using deprecated 'gym' package.")


def single_element_tensor_to_int(x: Tensor) -> int:
    return int(x)


def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy(force=True)


GYM_TO_PEARL_ACTION_SPACE = {
    "Discrete": DiscreteActionSpace,
    "Box": BoxActionSpace
    # Add more here as needed
}
PEARL_TO_GYM_ACTION = {
    "Discrete": single_element_tensor_to_int,
    "Box": tensor_to_numpy,
    # Add more here as needed
}


class GymEnvironment(Environment):
    """A wrapper for `gym.Env` to behave like Pearl's `Environment`."""

    def __init__(
        self, env_or_env_name: Union[gym.Env, str], *args: Any, **kwargs: Any
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
        self._action_space: ActionSpace = _get_pearl_action_space(
            gym_space_name=self.gym_space_name, gym_env=self.env
        )

    @property
    def action_space(self) -> ActionSpace:
        """Returns the Pearl action space for this environment."""
        return self._action_space

    @property
    def gym_space_name(self) -> str:
        """Returns the name of the underlying gym action space."""
        return self.env.action_space.__class__.__name__

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    def reset(self) -> Tuple[Observation, ActionSpace]:
        """Resets the environment and returns the initial observation and
        initial action space."""
        reset_result = self.env.reset()
        if isinstance(reset_result, Iterable) and isinstance(reset_result[1], dict):
            # newer Gym versions return an info dict.
            observation, info = reset_result
        else:
            # TODO: Deprecate this part at some point and only support new
            # version of Gymnasium?
            observation = list(reset_result.values())[0]  # pyre-ignore
        if isinstance(observation, np.ndarray):
            observation = observation.astype(np.float32)
        return observation, self.action_space

    def step(self, action: Action) -> ActionResult:
        """Takes one step in the environment given the agent's action. Returns an
        `ActionResult` object containing the next observation, reward, and done flag."""
        # Convert action to the format expected by Gymnasium
        effective_action = _get_gym_action(
            pearl_action=action, gym_space_name=self.gym_space_name
        )
        # Take a step in the environment and receive an action result
        gym_action_result = self.env.step(effective_action)
        if len(gym_action_result) == 4:
            # Older Gym versions use 'done' as opposed to 'terminated' and 'truncated'
            observation, reward, done, info = gym_action_result  # pyre-ignore
            terminated = done
            truncated = False
        elif len(gym_action_result) == 5:
            # Newer Gym versions use 'terminated' and 'truncated'
            observation, reward, terminated, truncated, info = gym_action_result
        else:
            raise ValueError(
                f"Unexpected action result from Gym (expected 4 or 5 elements): {gym_action_result}"
            )
        if isinstance(observation, np.ndarray):
            observation = observation.astype(np.float32)
        if isinstance(reward, np.float64):
            reward = reward.astype(np.float32)
        return ActionResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # pyre-fixme[3]: Return type must be annotated.
    def render(self):
        return self.env.render()

    def close(self) -> None:
        return self.env.close()

    def __str__(self) -> str:
        return self.env.spec.id


def _get_gym_action(
    pearl_action: Action, gym_space_name: str
) -> Union[int, np.ndarray]:
    """A helper function to convert a Pearl `Action` to an action compatible with
    the Gym environment specified by `gym_space_name`."""
    try:
        pearl_to_gym_action_transform = PEARL_TO_GYM_ACTION[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return pearl_to_gym_action_transform(pearl_action)


def _get_pearl_action_space(gym_space_name: str, gym_env: gym.Env) -> ActionSpace:
    """Returns the Pearl action space for this environment."""
    try:
        pearl_action_space_cls = GYM_TO_PEARL_ACTION_SPACE[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return pearl_action_space_cls.from_gym(gym_env.action_space)
