#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
This file contains helpers for unittest creation
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from pearl.api.action_space import ActionSpace
from pearl.replay_buffer.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffer.transition import TransitionBatch
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
    curr_available_actions = action_space
    done = torch.randint(2, (batch_size,)).float()

    (
        curr_available_actions_tensor_with_padding,
        curr_available_actions_mask,
    ) = TensorBasedReplayBuffer._create_action_tensor_and_mask(
        action_space, curr_available_actions
    )

    (
        next_available_actions_tensor_with_padding,
        next_available_actions_mask,
    ) = TensorBasedReplayBuffer._create_action_tensor_and_mask(
        action_space, next_available_actions
    )
    action_tensor = F.one_hot(actions, num_classes=action_space.n)
    batch = TransitionBatch(
        state=states,
        action=action_tensor,
        reward=rewards,
        next_state=next_states,
        next_action=action_tensor,
        curr_available_actions=curr_available_actions_tensor_with_padding.expand(
            batch_size, -1, -1
        ),
        curr_available_actions_mask=curr_available_actions_mask.expand(batch_size, -1),
        next_available_actions=next_available_actions_tensor_with_padding.expand(
            batch_size, -1, -1
        ),
        next_available_actions_mask=next_available_actions_mask.expand(batch_size, -1),
        done=done,
    )
    return batch, action_space


# for testing vanilla mlps
def create_normal_pdf_training_data(
    input_dim: int, num_data_points: int
) -> (torch.Tensor, torch.Tensor):

    """
    x are sampled from a multivariate normal distribution
    y are the corresponding pdf values
    """
    mean = torch.zeros(input_dim)
    sigma = torch.eye(input_dim)

    multi_variate_normal = torch.distributions.MultivariateNormal(mean, sigma)
    x = multi_variate_normal.sample(
        (num_data_points,)
    )  # sample from a mvn distribution
    y = torch.exp(
        -0.5 * ((x - mean) @ torch.inverse(sigma) * (x - mean)).sum(dim=1)
    ) / (
        torch.sqrt((2 * torch.tensor(3.14)) ** mean.shape[0] * torch.det(sigma))
    )  # corresponding pdf of mvn
    y_corrupted = y + 0.01 * torch.randn(num_data_points)  # noise corrupted targets
    return x, y_corrupted
