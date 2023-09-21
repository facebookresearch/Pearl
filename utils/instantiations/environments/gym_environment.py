from typing import Iterable

import gym
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

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the wrapper to be around a `gym.Env`.
        Args:
            args: positional arguments to be passed on to Gym's environment.
            kwargs: keyword arguments to be passed on to Gym's environment.
        """
        self.env = gym.make(*args, **kwargs)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self) -> (Observation, ActionSpace):
        reset_result = self.env.reset()
        if isinstance(reset_result, Iterable) and isinstance(reset_result[1], dict):
            # newer Gym versions return an info dict.
            observation, info = reset_result
        else:
            observation = reset_result
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
        return ActionResult(observation, reward, terminated, truncated, info)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self) -> str:
        return self.env.spec.id
