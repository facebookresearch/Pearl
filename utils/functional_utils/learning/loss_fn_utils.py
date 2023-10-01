#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import torch
from pearl.utils.functional_utils.learning.extend_state_feature import (
    extend_state_feature_by_available_action_space,
)
from torch import Tensor


# pyre-fixme[2]: Parameter must be annotated.
def compute_cql_loss(q_network, batch, batch_size) -> torch.Tensor:

    """
    Compute CQL loss for a batch of data.

    Inputs:
    1) q_network: to compute the q values of every (state, action) pair.
    2) batch: batch of data transitions (state, action, reward, done, next_state) along with (current and next)
            available actions.
    3) batch_size: size of batch.

    Outputs:
    cql_loss: Tensor with gradients.

    To compute cql_loss:
    1) Step 1: extend batch.state (2d tensor) with the available actions for each state to get a 3d tensor.
    2) Step 2: get q values of a batch of states and all corresponding available actions for each state.
    3) Step 3: get q values of (state, action) pairs in the batch.
    4) Step 4: compute cql_loss = 1/(batch_size) * (
                            sum_{state in batch} [log( sum_{action in current_available actions} exp(Q(state, action)) )]
                            - sum_{(state, action) in batch} Q(state, action)
                            ).
    Note: the first term in computing the cql loss uses state action values for all actions for each state in the batch
    while the second term only uses (state, action) in the batch.
    """

    # Step 1
    state_repeated_batch = extend_state_feature_by_available_action_space(
        state_batch=batch.state,
        curr_available_actions_batch=batch.curr_available_actions,
    )  # output shape: [batch_size, available_action_space_size, state_dim(dim of state features)]

    ## To Do: change the output shape of get_q_values method - .view(-1) should not be done in value_networks.py
    # Step 2
    q_values_state_all_available_actions = q_network.get_q_values(
        state_repeated_batch, batch.curr_available_actions
    ).view(
        batch_size, -1
    )  # shape: [batch_size, available_action_space_size]

    # Step 3
    q_values_state_actions_in_batch = q_values_state_all_available_actions.gather(
        1, batch.action
    )

    # Step 4
    cql_loss = (
        torch.logsumexp(q_values_state_all_available_actions, dim=-1).mean()
        - q_values_state_actions_in_batch.mean()
    )

    return cql_loss


# pyre-fixme[3]: Return type must be annotated.
def compute_elementwise_huber_loss(input_errors: Tensor, kappa: float = 1.0):
    huber_loss = torch.where(
        torch.abs(input_errors) <= kappa,
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        0.5 * (input_errors**2),
        kappa * (torch.abs(input_errors) - (0.5 * kappa)),
    )
    return huber_loss
