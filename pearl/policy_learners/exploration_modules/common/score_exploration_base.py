# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from __future__ import annotations
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
    """Base class for value-based exploration modules.

    Sub-classes implement :meth:`get_scores`, returning a preference score for
    each action in the provided :class:`ActionSpace`. Higher scores correspond
    to more preferred actions.

    Notes
    -----
    * ``get_scores`` must return a tensor with shape ``(batch_size, action_count)``.
    * For a single transition without a leading batch dimension, sub-classes
      should still return scores with shape ``(1, action_count)``; this method
      will then return a scalar action index.
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
                "exploit_action is deprecated. use `values` instead",
                DeprecationWarning,
            )
            return exploit_action
        
        if values is None:  # pragma: no cover - sanity check
            raise ValueError("`values` must be supplied for value-based exploration.")

        scores = self.get_scores(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=representation,
        )
        scores = assert_is_tensor_like(scores)

        action_index_batch = get_model_action_index_batch(scores, action_availability_mask)
        action_index_batch = action_index_batch.squeeze(-1)

        # If caller provided a single transition without a batch dimension, return a scalar rather than a size-1 tensor.
        if subjective_state.dim() == scores.dim() - 1:
            return action_index_batch.squeeze(0)

        return action_index_batch

    @abstractmethod
    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        *,
        representation: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Get the scores for each action.

        Args:
            subjective_state : torch.Tensor
                Shape ``(batch_size, feature_dim)`` or ``(feature_dim,)``.
            action_space : ActionSpace
                The discrete action space.
            values : torch.Tensor
                Value estimates with shape ``(batch_size, action_count)`` or ``(action_count,)``.
            representation : torch.nn.Module, optional
                A state representation network.
        Returns:
            return shape(batch_size, action_count)
        """
        raise NotImplementedError

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
