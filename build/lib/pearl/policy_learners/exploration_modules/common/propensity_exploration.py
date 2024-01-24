# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class PropensityExploration(ExplorationModule):
    """
    Propensity exploration module.
    """

    def __init__(self) -> None:
        super(PropensityExploration, self).__init__()

    # TODO: We should make discrete action space itself iterable
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        return torch.distributions.Categorical(values).sample()
