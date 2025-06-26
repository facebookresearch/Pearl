# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import Tensor


def argmax_random_tie_breaks(
    scores: Tensor, mask: Tensor | None = None
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


def argmax_random_tie_break_per_row(
    scores: Tensor, mask: Tensor | None = None, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Given a 2D tensor of scores, return the indices of the max score for each row.
    If there are ties inside a row, uniformly randomize among the ties.
    IMPORTANT IMPLEMENTATION DETAILS:
        1. Randomization is implemented independently for each row, unlike argmax_random_tie_breaks
           which uses the same permutation for all rows.
        2. Therefore this function is slower than argmax_random_tie_breaks

    Args:
        scores: A 2D tensor of scores of shape (batch_size, num_actions)
        mask [Optional]: A 2D score presence mask.
                         If missing, assuming that all scores are unmasked.
        epsilon: Threshold for considering scores as tied

    Returns:
        A 1D tensor of shape (batch_size,) containing the indices of the selected actions
    """
    # This function only works for 2D tensor
    assert scores.ndim == 2

    batch_size = scores.shape[0]

    # Find the maximum score in each row
    if mask is None:
        max_scores, _ = torch.max(scores, dim=1, keepdim=True)
        # Find actions that are within epsilon of the maximum score
        tied_actions = scores >= max_scores - epsilon
    else:
        # For masked case, we'll create a version of scores with -inf for masked values
        masked_scores = scores.clone()
        masked_scores[~mask.bool()] = float("-inf")

        # Now we can safely use torch.max
        max_scores, _ = torch.max(masked_scores, dim=1, keepdim=True)

        # Find actions that are within epsilon of the maximum score and are valid according to mask
        tied_actions = torch.logical_and(scores >= max_scores - epsilon, mask.bool())

    # For each example in the batch, randomly select one of the tied actions
    selected_actions = torch.zeros(batch_size, dtype=torch.long, device=scores.device)

    for i in range(batch_size):
        tied_indices = torch.nonzero(tied_actions[i]).squeeze()

        # If there's only one max action
        # tensor([[max_action_index]]).squeeze() becames scalar.
        if tied_indices.dim() == 0:
            selected_actions[i] = tied_indices.item()
        else:
            # Randomly select one of the tied actions
            random_idx = torch.randint(
                0, tied_indices.size(0), (1,), device=scores.device
            )
            selected_actions[i] = tied_indices[random_idx].item()

    return selected_actions  # Shape: (batch_size,)


def get_model_action_index_batch(
    scores: Tensor,
    mask: Tensor | None = None,
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
    """
    A helper function for concatenating all actions from a `DiscreteActionSpace`
    to a state or batch of states. The actions must be Tensors.

    This function uses the TorchScriptable version internally.

    Args:
        subjective_state: A Tensor of shape (batch_size, state_dim) or (state_dim).
        action_space: A `DiscreteActionSpace` object where each action is a Tensor.
        action_representation_module: Module to transform actions before concatenation.
        state_features_only: If True, only expand the state dimension without
            concatenating the actions.
    Returns:
        A Tensor of shape (batch_size, number_of_actions, state_dim + action_dim).
    """
    # Stack actions and apply action transformation
    raw_actions = torch.stack(action_space.actions).to(subjective_state.device)
    action_representations = action_representation_module(raw_actions)

    return concatenate_actions_to_state_scriptable(
        subjective_state=subjective_state,
        number_of_actions=action_space.n,
        action_representations=action_representations,
        state_features_only=state_features_only,
    )


def concatenate_actions_to_state_scriptable(
    subjective_state: Tensor,
    number_of_actions: int,
    action_representations: Tensor,
    state_features_only: bool = False,
) -> Tensor:
    """
    A TorchScriptable helper function for concatenating action representations
    to a state or batch of states.

    Args:
        subjective_state: A Tensor of shape (batch_size, state_dim) or (state_dim).
        number_of_actions: The number of actions (number_of_actions).
        action_representations: A Tensor of shape (number_of_actions, action_dim) containing
            all action representations.
        state_features_only: If True, only expand the state dimension without
            concatenating the action representations.
    Returns:
        A Tensor of shape (batch_size, number_of_actions, state_dim + action_dim).
    """
    state_dim = subjective_state.shape[-1]
    # Reshape to (batch_size, state_dim)
    subjective_state = subjective_state.view(-1, state_dim)
    batch_size = subjective_state.shape[0]
    action_dim = action_representations.shape[-1]

    # Expand to (batch_size, number_of_actions, state_dim) and return if `state_features_only`
    expanded_state = subjective_state.unsqueeze(1).repeat(1, number_of_actions, 1)
    if state_features_only:
        return expanded_state

    # Expand action representations to (batch_size, number_of_actions, action_dim)
    expanded_action_reps = action_representations.unsqueeze(0).repeat(batch_size, 1, 1)

    # (batch_size, number_of_actions, state_dim + action_dim)
    new_feature = torch.cat([expanded_state, expanded_action_reps], dim=2)
    torch._assert(
        new_feature.shape == (batch_size, number_of_actions, state_dim + action_dim),
        "The shape of the concatenated feature is wrong. Expected "
        f"{(batch_size, number_of_actions, state_dim + action_dim)}, got {new_feature.shape}",
    )
    return new_feature.to(subjective_state.device)
