# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional, Tuple

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation
from pearl.api.reward import Value
from pearl.api.space import Space
from pearl.utils.instantiations.environments.contextual_bandit_environment import (
    ContextualBanditEnvironment,
)
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace


class RewardIsEqualToTenTimesActionMultiArmBanditEnvironment(
    ContextualBanditEnvironment
):
    """
    A example implementation of a bandit environment. For simplicity, we assume
    no context. Therefore, it is a multi-arm bandit environment.
    """

    def __init__(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> Space | None:
        # For multi-arm bandit environments (where there are no 'observations'), we set the
        # observation space to be a DiscreteSpace with a single element, taken to be an empty
        # tensor.
        return DiscreteSpace([torch.tensor([])])

    def reset(self, seed: int | None = None) -> tuple[Observation, ActionSpace]:
        # Function returning the context and the available action space
        # Here, we use no context (None), but we could return varied implementations.
        return None, self.action_space

    def get_reward(self, action: Action) -> Value:
        # Here goes the code for computing the reward given an action on the current state
        # In this example, the reward is 10 times the digit representing the action.
        return (action * 10).item()

    def __str__(self) -> str:
        return "Bandit with reward = 10 * action index"
