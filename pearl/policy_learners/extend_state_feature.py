#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch


def extend_state_feature_by_available_action_space(
    state_batch: torch.Tensor,
    curr_available_actions_batch: torch.Tensor,
) -> torch.Tensor:
    """
    This is a helper function.
    What : It adds one more dimension (the action dimension) and extend 2d tensor to 3d tensor.
    How : It repeats state features by len(available_action) times.
    Return: (batch_size x actions x features), features of a single state with multi-actions
    Example: 1*4 --> 1*2*6 where 2 actions are added into dim=-2, where batch_size=1, state_dim=4, action_dim=2
        [1,2,3,4] -->
        [
            [1,2,3,4, 0,1]
            [1,2,3,4, 1,0]
        ]
    """
    # states features
    state_repeated_batch = torch.repeat_interleave(
        state_batch.unsqueeze(1),
        curr_available_actions_batch.shape[-2],  # num of actions
        dim=1,
    )  # (batch_size x action_space_size x state_dim)

    # actions & states features
    state_multi_actions_batch = torch.cat(
        [
            state_repeated_batch,
            curr_available_actions_batch,  # (batch_size, action_space_size, action_space_size)
        ],
        dim=-1,
    )  # (batch_size x action_space_size x (state_dim + action_dim))
    return state_multi_actions_batch
