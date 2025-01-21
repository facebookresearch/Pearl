# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class DoubleDQN(DeepQLearning):
    """
    Double DQN Policy Learner
    Compare to DQN, it gets a' from Q network and Q(s', a') from target network
    while DQN, get both a' and Q(s', a') from target network

    https://arxiv.org/pdf/1509.06461.pdf
    """

    @torch.no_grad()
    def get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        assert batch.next_state is not None

        assert batch.next_available_actions is not None
        assert batch.next_unavailable_actions_mask is not None

        assert isinstance(self._action_space, DiscreteActionSpace)

        next_state_action_values = self._Q.get_q_values(
            batch.next_state,  # (batch_size x state_dim)
            batch.next_available_actions,  # (batch_size x action_space_size x action_dim)
        )  # (batch_size x action_space_size)
        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[batch.next_unavailable_actions_mask] = -float("inf")

        # Torch.max(1) returns value, indices
        next_action_indices = next_state_action_values.max(1)[1]  # (batch_size)
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        next_action_batch = batch.next_available_actions[
            # pyre-fixme[16]: Optional type has no attribute `size`.
            torch.arange(batch.next_available_actions.size(0)),
            next_action_indices.squeeze(),
        ]  # (batch_size x action_dim)
        return self._Q_target.get_q_values(
            # pyre-fixme[6]: expected `Tensor` but got `Optional[Tensor]`
            batch.next_state,
            next_action_batch,
        )  # (batch_size)

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two DoubleDQN instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        # First, perform the comparisons from the base class
        differences.append(super().compare(other))

        if not isinstance(other, DoubleDQN):
            differences.append("other is not an instance of DoubleDQN")

        # No additional attributes to compare in DoubleDQN

        return "\n".join(differences)
