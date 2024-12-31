# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import abstractmethod
from typing import List
from warnings import warn

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
    ExplorationType,
)
from pearl.utils.functional_utils.learning.action_utils import (
    get_model_action_index_batch,
)
from pearl.utils.tensor_like import assert_is_tensor_like


class ScoreExplorationBase(ExplorationModule):
    """
    Value exploration base module.
    Specific exploration module subclasses need to implement `get_scores`.
    Actions with highest scores will be chosen.
    """

    def __init__(self) -> None:
        super().__init__()
        self.exploration_type: ExplorationType = ExplorationType.VALUE

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        action_availability_mask: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        """
        Args:
            subjective_state is in shape of (batch_size, feature_size) or (feature_size)
            for a single transition
            values is in shape of (batch_size, action_count) or (action_count)
        Returns:
            return shape(batch_size,)
        """
        if exploit_action is not None:
            warn(
                "exploit_action shouldn't be used. use `values` instead",
                DeprecationWarning,
            )
            return exploit_action
        assert values is not None
        scores = self.get_scores(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=representation,
        )  # shape: (batch_size, action_count)
        scores = assert_is_tensor_like(scores)
        action_index_batch = get_model_action_index_batch(
            scores, action_availability_mask
        )
        return action_index_batch.squeeze(-1)
        # FIXME: the squeeze(-1) is a hack.
        # It is used to get rid of the batch dimension if the batch has a
        # single element. For example, if action_index_batch is
        # torch.tensor([0]), then the result will be the batch-less index 0.
        # The rationale is that if the batch has a single element, then
        # subject_state was batchless and self.get_score introduced a batch
        # dimension (for uniformity and convenience of operations, which can
        # then all assume batch form), so the batch dimension should be removed.
        # The problem with this approach is that it is heuristic and not
        # correct in all cases. For example, if subject_state is *not* batchless
        # but has a single element, then the returned value should be a
        # single-element batch containing one index, but in this case
        # squeeze will incorrectly remove the batch dimension.
        # The correct approach should be that all functions manipulate tensors
        # in the same way PyTorch modules do, namely accepting input that
        # may have a batch dimension or not, and have all following tensors
        # mirroring that.

    @abstractmethod
    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        exploit_action: Action | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        """
        Get the scores for each action.

        Args:
            subjective_state is in shape of (batch_size, feature_size) or (feature_size)
            for a single transition
            values is in shape of (batch_size, action_count) or (action_count)
        Returns:
            return shape(batch_size, action_count)
        """
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

        if not isinstance(other, ScoreExplorationBase):
            differences.append("other is not an instance of ScoreExplorationBase")
        else:
            if self.exploration_type != other.exploration_type:
                differences.append(
                    f"exploration_type is different: {self.exploration_type} "
                    + f"vs {other.exploration_type}"
                )

        return "\n".join(differences)
