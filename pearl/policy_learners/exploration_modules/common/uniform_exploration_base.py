# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import abstractmethod

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
    ExplorationType,
)


class UniformExplorationBase(ExplorationModule):
    """
    Uniform exploration module.
    """

    def __init__(self) -> None:
        self.exploration_type: ExplorationType = ExplorationType.UNIFORM

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action | None = None,
        values: torch.Tensor | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        pass
