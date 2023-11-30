from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation


class Environment(ABC):
    """An abstract interface for environments. An `Environment` is an object
    that the agent interacts with and provides the agent with observations in
    the form of an `ActionResult` object. This interface follows the design of
    environments in Gymnasium.
    """

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """Returns the action space of the environment."""
        pass

    @abstractmethod
    def reset(self) -> Tuple[Observation, ActionSpace]:
        """Resets the environment and returns the initial observation and
        initial available action space."""
        pass

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        """Takes one step in the environment given the agent's action. Returns an
        `ActionResult` object containing the next observation, reward, and done flag."""
        pass

    # TODO: Properly handle typing for `render()`.
    def render(self):  # pyre-ignore[3] # noqa: B027
        """Renders the environment. Default implementation does nothing."""
        pass

    def close(self) -> None:  # noqa: B027
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        pass
