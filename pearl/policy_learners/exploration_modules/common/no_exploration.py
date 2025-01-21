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
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules import ExplorationModule
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace


class NoExploration(ScoreExplorationBase):
    """
    An exploration module that does not explore.
    It implements a `get_score` function that assumes `values` is given
    and simply returns the values for each action.
    """

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        if exploit_action is not None:
            raise ValueError("exploit_action shouldn't be used. use `values` instead")
        assert isinstance(action_space, DiscreteSpace)
        assert values is not None
        return values.view(-1, action_space.n)  # batch_size, action_count

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two NoExploration instances for equality.

        Since this module has no attributes or buffers to compare,
        it only checks if the `other` object is an instance of the same class.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, NoExploration):
            differences.append("other is not an instance of NoExploration")

        return "\n".join(differences)
