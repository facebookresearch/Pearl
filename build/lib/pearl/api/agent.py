# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation


class Agent(ABC):
    """
    An abstract interface for agents.
    """

    @abstractmethod
    def act(self) -> Action:
        pass

    @abstractmethod
    def observe(self, action_result: ActionResult) -> None:
        pass

    @abstractmethod
    def learn(self) -> object:
        pass

    @abstractmethod
    def reset(
        self, observation: Observation, available_action_space: ActionSpace
    ) -> None:
        pass
