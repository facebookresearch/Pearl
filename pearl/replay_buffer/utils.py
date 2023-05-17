#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn.functional as F
from pearl.api.action_space import ActionSpace


def create_next_action_tensor_and_mask(
    action_space: ActionSpace, next_available_actions: ActionSpace
) -> (torch.tensor, torch.tensor):
    next_available_actions_tensor_with_padding = torch.zeros(
        (1, action_space.n, action_space.n)
    )  # (1 x action_space_size x action_dim)
    next_available_actions_tensor = F.one_hot(
        torch.arange(0, next_available_actions.n), num_classes=action_space.n
    )  # (1 x available_action_space_size x action_dim)
    next_available_actions_tensor_with_padding[
        0, : next_available_actions.n, :
    ] = next_available_actions_tensor
    next_available_actions_mask = torch.zeros(
        (1, action_space.n)
    )  # (1 x action_space_size)
    next_available_actions_mask[0, next_available_actions.n :] = 1
    next_available_actions_mask = next_available_actions_mask.bool()

    return (next_available_actions_tensor_with_padding, next_available_actions_mask)
