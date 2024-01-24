# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

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

    # FIXME: add this and in implement in all concrete subclasses
    # @property
    # @abstractmethod
    # def observation_space(self) -> Space:
    #     """Returns the observation space of the environment."""
    #     pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        """Resets the environment and returns the initial observation and
        initial available action space."""
        pass

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        """Takes one step in the environment given the agent's action. Returns an
        `ActionResult` object containing the next observation, reward, and done flag."""
        pass

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None
