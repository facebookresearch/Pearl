#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
This file contains helpers for unittest creation
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from pearl.api.action_space import ActionSpace
from pearl.replay_buffer.transition import TransitionBatch
from pearl.replay_buffer.utils import create_next_action_tensor_and_mask
from pearl.utils.action_spaces import DiscreteActionSpace


def create_random_batch(
    action_dim: int, state_dim: int, batch_size: int
) -> Tuple[TransitionBatch, ActionSpace]:
    states = torch.rand(batch_size, state_dim)
    actions = torch.randint(action_dim, (batch_size,))
    rewards = torch.rand(
        batch_size,
    )
    next_states = torch.rand(batch_size, state_dim)
    action_space = DiscreteActionSpace(range(action_dim))
    next_available_actions = action_space
    done = torch.randint(2, (batch_size,)).float()

    (
        next_available_actions_tensor_with_padding,
        next_available_actions_mask,
    ) = create_next_action_tensor_and_mask(action_space, next_available_actions)
    action_tensor = F.one_hot(actions, num_classes=action_space.n)
    batch = TransitionBatch(
        state=states,
        action=action_tensor,
        reward=rewards,
        next_state=next_states,
        next_available_actions=next_available_actions_tensor_with_padding.expand(
            batch_size, -1, -1
        ),
        next_available_actions_mask=next_available_actions_mask.expand(batch_size, -1),
        done=done,
    )
    return batch, action_space
