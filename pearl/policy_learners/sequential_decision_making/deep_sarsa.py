# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action_space import ActionSpace
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.deep_td_learning import (
    DeepTDLearning,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


class DeepSARSA(DeepTDLearning):
    """
    A Deep Temporal Difference learning policy learner.
    """

    def __init__(
        self,
        state_dim: int | None = None,
        action_space: ActionSpace | None = None,
        exploration_module: ExplorationModule | None = None,
        action_representation_module: ActionRepresentationModule | None = None,
        optimizer: Optional[optim.Optimizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else EGreedyExploration(0.05)
            ),
            on_policy=True,
            action_representation_module=action_representation_module,
            optimizer=optimizer,
            **kwargs,
        )

    @torch.no_grad()
    def get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        """
        For SARSA, next state values comes from committed next action + next state value
        """
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_action_batch = batch.next_action  # (batch_size x action_dim)
        assert next_state_batch is not None, "SARSA needs to have next state"
        assert next_action_batch is not None, "SARSA needs to have next action"

        # use get_batch method instead of doing forward pass
        next_state_action_values = self._Q_target.get_q_values(
            next_state_batch, next_action_batch
        )  # (batch_size)

        return next_state_action_values

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two DeepSARSA instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        # Inherit comparisons from the base class
        differences.append(super().compare(other))

        if not isinstance(other, DeepSARSA):
            differences.append("other is not an instance of DeepSARSA")

        # No additional attributes to compare in DeepSARSA

        return "\n".join(differences)
