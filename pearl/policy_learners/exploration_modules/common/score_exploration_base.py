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
from pearl.policy_learners.exploration_modules.common.tiebreaking_strategy import (
    TiebreakingStrategy,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
    ExplorationType,
)
from pearl.utils.functional_utils.learning.action_utils import (
    get_model_action_index_batch,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.tensor_like import assert_is_tensor_like


class ScoreExplorationBase(ExplorationModule):
    """
    Value exploration base module.
    Specific exploration module subclasses need to implement `get_scores`.
    Actions with highest scores will be chosen, with a random tie-breaking
    if that option is selected.
    """

    def __init__(
        self,
        randomized_tiebreaking: TiebreakingStrategy = TiebreakingStrategy.NO_TIEBREAKING,
    ) -> None:
        super().__init__()
        self.exploration_type: ExplorationType = ExplorationType.VALUE
        self.randomized_tiebreaking = randomized_tiebreaking

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
            subjective_state is in shape of (batch_size, action_count, feature_size)
            or (action_count, feature_size) for a single transition
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

        assert isinstance(action_space, DiscreteActionSpace)
        assert subjective_state.ndim in {2, 3}

        # TODO: commenting out assertion becase NeuralLinearBandit
        # seems to violate it in this line:
        # subjective_state=model_ret["nn_output"]. Fix this.
        # assert subjective_state.shape[-2] == action_space.n

        assert values is not None
        assert values.ndim in {1, 2}
        assert values.shape[-1] == action_space.n

        scores = self.get_scores(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=representation,
        )  # shape: (batch_size, action_count)
        scores = assert_is_tensor_like(scores)
        action_index_batch = get_model_action_index_batch(
            scores,
            action_availability_mask,
            self.randomized_tiebreaking,
        )

        actions = torch.nn.functional.embedding(
            action_index_batch, action_space.actions_batch
        )
        return actions

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
        Compares two ScoreExplorationBase instances for equality.

        Checks if the other object is an instance of the same class and compares
        the exploration_type and randomized_tiebreaking attributes.

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

            if self.randomized_tiebreaking != other.randomized_tiebreaking:
                differences.append(
                    f"randomized_tiebreaking is different: {self.randomized_tiebreaking} "
                    + f"vs {other.randomized_tiebreaking}"
                )

        return "\n".join(differences)
