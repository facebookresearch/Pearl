# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random
from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.uniform_exploration_base import (
    UniformExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class EGreedyExploration(UniformExplorationBase):
    """
    epsilon Greedy exploration module.
    """

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action | None,
        values: torch.Tensor | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        if exploit_action is None:
            raise ValueError(
                "exploit_action cannot be None for epsilon-greedy exploration"
            )
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("action space must be discrete")
        if random.random() < self.epsilon:
            return action_space.sample(action_availability_mask).to(
                exploit_action.device
            )
        else:
            return exploit_action
