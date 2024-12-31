# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

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
        super().__init__()

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("action space must be discrete")
        action_index = torch.distributions.Categorical(values).sample()
        return action_space.actions[action_index]

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two PropensityExploration instances for equality.

        Since this module has no attributes or buffers to compare,
        it only checks if the `other` object is an instance of the same class.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, PropensityExploration):
            differences.append("other is not an instance of PropensityExploration")

        return "\n".join(differences)
