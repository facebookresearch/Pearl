# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import abstractmethod
from typing import Optional

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
        exploit_action: Optional[Action] = None,
        values: Optional[torch.Tensor] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        pass
