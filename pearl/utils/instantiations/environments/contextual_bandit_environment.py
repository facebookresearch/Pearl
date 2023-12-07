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

from pearl.api.environment import Environment
from pearl.api.reward import Reward


class ContextualBanditEnvironment(Environment, ABC):
    """
    A specialization of Environment for contextual bandits.
    In a contextual bandit environment, an episode always has a single step
    and the only important information the environment needs to produce
    as a result of an action is its reward.

    This class provides an implementation of step that takes that into account,
    returning an ActionResult with 'terminated' equal to 'True' and with next observation
    equal to None (since it is irrelevant).
    It defers to a new method `get_reward` (to be provided by implementations)
    to determine the ActionResult reward.
    """

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        pass

    @abstractmethod
    def get_reward(self, action: Action) -> Reward:
        pass

    def step(self, action: Action) -> ActionResult:
        # Since all episodes have a single step,
        # the resulting observation after an action does not matter,
        # so we set it to None.
        reward = self.get_reward(action)
        return ActionResult(
            observation=None,
            reward=reward,
            terminated=True,
            truncated=False,
        )

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
