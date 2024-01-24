# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def extend_state_feature_by_available_action_space(
    state_batch: torch.Tensor,
    curr_available_actions_batch: torch.Tensor,
) -> torch.Tensor:
    """
    This is a helper function.

    Input dim:
    state_batch: batch_size x state_dim
    curr_available_actions_batch: batch_size x available_action_space_size x action_dim

    Output dim:
    state_available_actions_batch: batch_size x available_action_space_size x state_dim
    """

    state_repeated_batch = torch.repeat_interleave(
        state_batch.unsqueeze(1),
        curr_available_actions_batch.shape[-2],  # num of available actions
        dim=1,
    )  # (batch_size x available_action_space_size x state_dim)

    """
    The above step adds one more dimension (number of available actions) and extends state_batch
    which is a 2d tensor of shape (batch_size x state_dim) to 3d tensor of shape
    (batch_size x available_action_space_size x state_dim).

    How: it adds a dimension and repeats state features available_action_space_size times

    Example:
    state_batch: [1 x 4] --> state_repeated_batch: [1 x 2 x 4], where batch_size=1, state_dim=4,
    and available_action_space_size=2. Hence, 2 actions are added into dim=-2
    [[1,2,3,4]] --> [[[1,2,3,4],
                      [1,2,3,4]]]
    """

    return state_repeated_batch
