# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import abstractmethod
from typing import List

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
        super().__init__()
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

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two UniformExplorationBase instances for equality.

        Since this module has no attributes or buffers to compare,
        it only checks if the `other` object is an instance of the same class.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        if not isinstance(other, UniformExplorationBase):
            differences.append("other is not an instance of UniformExplorationBase")
        else:
            if self.exploration_type != other.exploration_type:
                differences.append(
                    f"exploration_type is different: {self.exploration_type} "
                    + f"vs {other.exploration_type}"
                )

        return "\n".join(differences)
