# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import Tensor


def argmax_random_tie_breaks(
    scores: Tensor, mask: Optional[Tensor] = None
) -> torch.Tensor:
    """
    Given a 2D tensor of scores, return the indices of the max score for each row.
    If there are ties inside a row, uniformly randomize among the ties.
    IMPORTANT IMPLEMENTATION DETAILS:
        1. Randomization is implemented consistently across all rows. E.g. if several columns
            are tied on 2 different rows, we will return the same index for each of these rows.

    Args:
        scores: A 2D tensor of scores
        mask [Optional]: A 2D score presence mask.
                         If missing, assuming that all scores are unmasked.
    """
    # This function only works for 2D tensor
    assert scores.ndim == 2

    # Permute the columns
    num_cols = scores.size(1)
    random_col_indices = torch.randperm(num_cols)
    permuted_scores = torch.index_select(scores, 1, random_col_indices)
    if mask is not None:
        permuted_mask = torch.index_select(mask, 1, random_col_indices)
        permuted_scores = torch.masked.as_masked_tensor(
            permuted_scores, permuted_mask.bool()
        )

    # Find the indices of the maximum elements in the random permutation
    max_indices_in_permuted_data = torch.argmax(permuted_scores, dim=1)

    if mask is not None:
        # pyre-fixme[16]: `Tensor` has no attribute `get_data`.
        max_indices_in_permuted_data = max_indices_in_permuted_data.get_data().long()

    # Use the random permutation to get the original indices of the maximum elements
    argmax_indices = random_col_indices[max_indices_in_permuted_data]

    return argmax_indices


def get_model_actions(
    scores: Tensor,
    mask: Optional[Tensor] = None,
    randomize_ties: bool = False,
) -> torch.Tensor:
    """
    Given a tensor of scores, get the indices of chosen actions.
    Chosen actions are the score argmax (within each row), subject to optional mask.
    if `randomize_ties`=True, we will also randomize the order of tied actions with
        maximum values. This has computational cost compared to not randomizing (use 1st index)

    Args:
        scores: A 2D tensor of scores
        mask [Optional]: A 2D score presence mask.
                         If missing, assuming that all scores are unmasked.

    Returns:
        1D tensor of size (batch_size,)
    """
    if randomize_ties:
        model_actions = argmax_random_tie_breaks(scores, mask)
    else:
        if mask is None:
            # vanilla argmax - no masking or randomization
            model_actions = torch.argmax(scores, dim=1)
        else:
            # mask out non-present arms
            scores_masked = torch.masked.as_masked_tensor(scores, mask.bool())
            model_actions = (
                # pyre-fixme[16]: `Tensor` has no attribute `get_data`.
                torch.argmax(scores_masked, dim=1).get_data()
            )
    return model_actions


def concatenate_actions_to_state(
    subjective_state: Tensor,
    action_space: DiscreteActionSpace,
    action_representation_module: ActionRepresentationModule,
    state_features_only: bool = False,
) -> Tensor:
    """A helper function for concatenating all actions from a `DiscreteActionSpace`
    to a state or batch of states. The actions must be Tensors.

    Args:
        subjective_state: A Tensor of shape (batch_size, state_dim) or (state_dim).
        action_space: A `DiscreteActionSpace` object where each action is a Tensor.
        state_features_only: If True, only expand the state dimension without
            concatenating the actions.
    Returns:
        A Tensor of shape (batch_size, action_count, state_dim + action_dim).
    """
    state_dim = subjective_state.shape[-1]
    # Reshape to (batch_size, state_dim)
    subjective_state = subjective_state.view(-1, state_dim)
    batch_size = subjective_state.shape[0]

    # action dim is the dimension of the output of action representation if set
    if action_representation_module.representation_dim != -1:
        action_dim = action_representation_module.representation_dim
    else:
        action_dim = action_space.action_dim
    action_count = action_space.n

    # Expand to (batch_size, action_count, state_dim) and return if `state_features_only`
    expanded_state = subjective_state.unsqueeze(1).repeat(1, action_count, 1)
    if state_features_only:
        return expanded_state

    # Stack actions and expand to (batch_size, action_count, action_dim)
    actions = torch.stack(action_space.actions).to(subjective_state.device)
    # Apply action transformation (default is the identity transformation)
    actions = action_representation_module(actions)
    expanded_action = actions.unsqueeze(0).repeat(batch_size, 1, 1)

    # (batch_size, action_count, state_dim + action_dim)
    new_feature = torch.cat([expanded_state, expanded_action], dim=2)
    torch._assert(
        new_feature.shape == (batch_size, action_count, state_dim + action_dim),
        "The shape of the concatenated feature is wrong. Expected "
        f"{(batch_size, action_count, state_dim + action_dim)}, got {new_feature.shape}",
    )
    return new_feature.to(subjective_state.device)
