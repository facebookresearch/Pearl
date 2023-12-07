# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
        super(EGreedyExploration, self).__init__()
        self.epsilon = epsilon

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Optional[Action],
        values: Optional[torch.Tensor] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        if exploit_action is None:
            raise ValueError(
                "exploit_action cannot be None for epsilon-greedy exploration"
            )
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("action space must be discrete")
        if random.random() < self.epsilon:
            return torch.randint(action_space.n, (1,))
        return exploit_action
