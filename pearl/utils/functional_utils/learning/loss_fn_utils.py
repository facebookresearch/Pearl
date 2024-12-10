# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import torch
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import Tensor


def compute_cql_loss(
    q_network: QValueNetwork, batch: TransitionBatch, batch_size: int
) -> torch.Tensor:
    """
    Compute CQL loss for a batch of data.

    Inputs:
    1) q_network: to compute the q values of every (state, action) pair.
    2) batch: batch of data transitions (state, action, reward, terminated, truncated, next_state)
            along with (current and next) available actions.
    3) batch_size: size of batch.

    Outputs:
    cql_loss: Tensor with gradients.

    To compute cql_loss:
    1) Step 1: extend batch.state (2d tensor) with the available actions for each state to get a
               3d tensor.
    2) Step 2: get q values of a batch of states and all corresponding available actions
               for each state.
    3) Step 3: get q values of (state, action) pairs in the batch.
    4) Step 4: compute cql_loss = 1/(batch_size) * (
                            sum_{state in batch}
                                  [log( sum_{action in current_available actions}
                                                   exp(Q(state, action)) )
                                  ]
                            - sum_{(state, action) in batch} Q(state, action)
                            ).
    Note: the first term in computing the cql loss uses state action values for all actions
    for each state in the batch while the second term only uses (state, action) in the batch.
    """
    assert batch.curr_available_actions is not None

    # TODO: change the output shape of get_q_values method - .view(-1) should not be done in
    # value_networks.py

    # Step 1
    assert batch.curr_available_actions is not None
    q_values_state_all_available_actions = q_network.get_q_values(
        batch.state, batch.curr_available_actions
    ).view(batch_size, -1)
    # shape: [batch_size, available_action_space_size]

    # Step 2
    q_values_state_actions_in_batch = q_values_state_all_available_actions.gather(
        1, batch.action.long()
    )

    # Step 3
    cql_loss = (
        torch.logsumexp(q_values_state_all_available_actions, dim=-1).mean()
        - q_values_state_actions_in_batch.mean()
    )

    return cql_loss


def compute_elementwise_huber_loss(input_errors: Tensor, kappa: float = 1.0) -> Tensor:
    huber_loss = torch.where(
        torch.abs(input_errors) <= kappa,
        0.5 * (input_errors.pow(2)),
        kappa * (torch.abs(input_errors) - (0.5 * kappa)),
    )
    return huber_loss
