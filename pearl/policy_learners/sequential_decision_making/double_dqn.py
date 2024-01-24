# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
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
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        assert next_state_batch is not None

        next_available_actions_batch = batch.next_available_actions
        assert next_available_actions_batch is not None

        # (batch_size x action_space_size x action_dim)
        next_unavailable_actions_mask_batch = (
            batch.next_unavailable_actions_mask
        )  # (batch_size x action_space_size)

        assert isinstance(self._action_space, DiscreteActionSpace)
        number_of_actions = self._action_space.n

        next_state_batch_repeated = torch.repeat_interleave(
            next_state_batch.unsqueeze(1),
            number_of_actions,
            dim=1,
        )  # (batch_size x action_space_size x state_dim)

        next_state_action_values = self._Q.get_q_values(
            next_state_batch_repeated, next_available_actions_batch
        ).view(
            (batch_size, -1)
        )  # (batch_size x action_space_size)
        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_unavailable_actions_mask_batch] = -float("inf")

        # Torch.max(1) returns value, indices
        next_action_indices = next_state_action_values.max(1)[1]  # (batch_size)
        next_action_batch = next_available_actions_batch[
            torch.arange(next_available_actions_batch.size(0)),
            next_action_indices.squeeze(),
        ]
        return self._Q_target.get_q_values(next_state_batch, next_action_batch)
