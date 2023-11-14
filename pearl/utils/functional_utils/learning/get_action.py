from typing import Optional

import torch


def argmax_random_tie_breaks(
    scores: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Given a 2D tensor of scores, return the indices of the max score for each row.
    If there are ties inside a row, uniformly randomize among the ties.
    IMPORTANT IMPLEMENTATION DETAILS:
        1. Randomization is implemented consistently across all rows. E.g. if several columns
            are tied on 2 different rows, we will return the same index for each of these rows.

    Args:
        scores: A 2D tensor of scores
        mask [Optional]: A 2D score presence mask. If missing, assuming that all scores are unmasked.
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
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    randomize_ties: bool = False,
) -> torch.Tensor:
    """
    Given a tensor of scores, get the indices of chosen actions.
    Chosen actions are the score argmax (within each row), subject to optional mask.
    if `randomize_ties`=True, we will also randomize the order of tied actions with
        maximum values. This has computational cost compared to not randomizing (use 1st index)

    Args:
        scores: A 2D tensor of scores
        mask [Optional]: A 2D score presence mask. If missing, assuming that all scores are unmasked.

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
