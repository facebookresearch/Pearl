# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.policy_learners.exploration_modules.common.tiebreaking_strategy import (
    TiebreakingStrategy,
)
from pearl.utils.functional_utils.learning.action_utils import (
    argmax_random_tie_break_per_row,
    argmax_random_tie_breaks_batch,
    concatenate_actions_to_state,
    concatenate_actions_to_state_scriptable,
    get_model_action_index_batch,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


@torch.jit.script
def concatenate_actions_to_state_script(
    subjective_state: torch.Tensor,
    number_of_actions: int,
    action_representations: torch.Tensor,
    state_features_only: bool = False,
) -> torch.Tensor:
    """TorchScript version of concatenate_actions_to_state"""
    return concatenate_actions_to_state_scriptable(
        subjective_state=subjective_state,
        number_of_actions=number_of_actions,
        action_representations=action_representations,
        state_features_only=state_features_only,
    )


class TestConcatenateActionsToState(unittest.TestCase):
    def test_concatenate_actions_to_state_scriptable(self) -> None:
        # Create test inputs
        batch_size = 2
        state_dim = 3
        number_of_actions = 4
        action_dim = 5

        # Create a sample state tensor
        subjective_state = torch.randn(batch_size, state_dim)

        # Create sample action representations
        action_representations = torch.randn(number_of_actions, action_dim)

        # Test the scriptable function
        result = concatenate_actions_to_state_script(
            subjective_state=subjective_state,
            number_of_actions=number_of_actions,
            action_representations=action_representations,
            state_features_only=False,
        )

        # Check the shape of the result
        self.assertEqual(
            result.shape, (batch_size, number_of_actions, state_dim + action_dim)
        )

        # Test with state_features_only=True
        result_state_only = concatenate_actions_to_state_script(
            subjective_state=subjective_state,
            number_of_actions=number_of_actions,
            action_representations=action_representations,
            state_features_only=True,
        )

        # Check the shape of the result
        self.assertEqual(
            result_state_only.shape, (batch_size, number_of_actions, state_dim)
        )

    def test_concatenate_actions_to_state(self) -> None:
        # Create test inputs
        batch_size = 2
        state_dim = 3
        number_of_actions = 4
        action_dim = 5

        # Create a sample state tensor
        subjective_state = torch.randn(batch_size, state_dim)

        # Create a sample action space
        actions = [torch.randn(action_dim) for _ in range(number_of_actions)]
        action_space = DiscreteActionSpace(actions)

        action_representation_module = IdentityActionRepresentationModule()

        # Test the function
        result = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            action_representation_module=action_representation_module,
            state_features_only=False,
        )

        # Check the shape of the result
        self.assertEqual(
            result.shape, (batch_size, number_of_actions, state_dim + action_dim)
        )

        # Test with state_features_only=True
        result_state_only = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            action_representation_module=action_representation_module,
            state_features_only=True,
        )

        # Check the shape of the result
        self.assertEqual(
            result_state_only.shape, (batch_size, number_of_actions, state_dim)
        )

    def test_functions_produce_same_output(self) -> None:
        # Create test inputs
        batch_size = 2
        state_dim = 3
        number_of_actions = 4
        action_dim = 5

        # Create a sample state tensor
        subjective_state = torch.randn(batch_size, state_dim)

        # Create a sample action space with fixed values for testing
        actions = [torch.ones(action_dim) * i for i in range(number_of_actions)]
        action_space = DiscreteActionSpace(actions)

        action_representation_module = IdentityActionRepresentationModule()

        # Get the result from the original function
        result_original = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            action_representation_module=action_representation_module,
            state_features_only=False,
        )

        # Stack actions and apply action transformation
        raw_actions = torch.stack(action_space.actions).to(subjective_state.device)
        action_representations = action_representation_module(raw_actions)

        # Get the result from the scriptable function
        result_scriptable = concatenate_actions_to_state_script(
            subjective_state=subjective_state,
            number_of_actions=number_of_actions,
            action_representations=action_representations,
            state_features_only=False,
        )

        # Check that both functions produce the same output
        self.assertTrue(torch.allclose(result_original, result_scriptable))

        # Test with state_features_only=True
        result_original_state_only = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            action_representation_module=action_representation_module,
            state_features_only=True,
        )

        result_scriptable_state_only = concatenate_actions_to_state_script(
            subjective_state=subjective_state,
            number_of_actions=number_of_actions,
            action_representations=action_representations,
            state_features_only=True,
        )

        # Check that both functions produce the same output with state_features_only=True
        self.assertTrue(
            torch.allclose(result_original_state_only, result_scriptable_state_only)
        )


class TestArgmaxRandomTieBreakPerRow(unittest.TestCase):
    def test_argmax_random_tie_break_per_row_no_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_break_per_row(scores)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(argmax_values_returned[0], {1, 2})
        self.assertSetEqual(argmax_values_returned[1], {0, 1})
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(argmax_values_returned[3], {1, 2})

    def test_argmax_random_tie_break_per_row_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_break_per_row(scores, mask)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_independent_randomization(self) -> None:
        """Test that randomization is independent for each row."""
        # Create a tensor with identical rows, each having multiple tied maximum values
        scores = torch.tensor(
            [
                [1, 10, 10, 10],
                [1, 10, 10, 10],
                [1, 10, 10, 10],
                [1, 10, 10, 10],
            ]
        )

        # Run the function many times and count how often each row selects the same index
        num_trials = 1000
        same_selection_count = 0

        for _ in range(num_trials):
            argmax = argmax_random_tie_break_per_row(scores)

            # Check if all rows selected the same index
            if torch.all(argmax == argmax[0]):
                same_selection_count += 1

        # Calculate the probability of all rows selecting the same index
        same_selection_probability = same_selection_count / num_trials

        # With independent randomization, the probability of all 4 rows selecting
        # the same index should be approximately (1/3)^3 ≈ 0.037
        # (since each row has 3 tied maximum values)
        # We allow some margin for random variation
        expected_probability = (1 / 3) ** 3
        self.assertLess(abs(same_selection_probability - expected_probability), 0.05)

        # For comparison, with the original argmax_random_tie_breaks_batch function,
        # this probability would be close to 1.0 since it uses the same permutation
        # for all rows


class TestGetAction(unittest.TestCase):
    def test_argmax_random_tie_breaks_batch_no_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks_batch(scores)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(argmax_values_returned[0], {1, 2})
        self.assertSetEqual(argmax_values_returned[1], {0, 1})
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(argmax_values_returned[3], {1, 2})

    def test_argmax_random_tie_breaks_batch_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks_batch(scores, mask)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_randomize(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_action_index_batch(
                scores, mask, tiebreaking_strategy=TiebreakingStrategy.BATCH_TIEBREAKING
            )
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_not_randomize(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_action_index_batch(
                scores, mask, tiebreaking_strategy=TiebreakingStrategy.NO_TIEBREAKING
            )
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[2],
            {
                0,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_per_row_randomize(self) -> None:
        """Test that PER_ROW_TIEBREAKING randomizes ties independently for each row."""
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_action_index_batch(
                scores,
                mask,
                tiebreaking_strategy=TiebreakingStrategy.PER_ROW_TIEBREAKING,
            )
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_tiebreaking_strategies_comparison(self) -> None:
        """Test and compare the behavior of different tiebreaking strategies."""
        # Create a tensor with identical rows, each having multiple tied maximum values
        scores = torch.tensor(
            [
                [1, 10, 10, 10],
                [1, 10, 10, 10],
                [1, 10, 10, 10],
                [1, 10, 10, 10],
            ]
        )

        # Test NO_TIEBREAKING
        argmax_no_tiebreaking = get_model_action_index_batch(
            scores, tiebreaking_strategy=TiebreakingStrategy.NO_TIEBREAKING
        )
        # With NO_TIEBREAKING, all rows should select the first maximum value (index 1)
        self.assertTrue(torch.all(argmax_no_tiebreaking == 1))

        # Test BATCH_TIEBREAKING vs PER_ROW_TIEBREAKING
        num_trials = 1000
        batch_same_selection_count = 0
        per_row_same_selection_count = 0

        for _ in range(num_trials):
            # BATCH_TIEBREAKING
            argmax_batch = get_model_action_index_batch(
                scores, tiebreaking_strategy=TiebreakingStrategy.BATCH_TIEBREAKING
            )
            if torch.all(argmax_batch == argmax_batch[0]):
                batch_same_selection_count += 1

            # PER_ROW_TIEBREAKING
            argmax_per_row = get_model_action_index_batch(
                scores, tiebreaking_strategy=TiebreakingStrategy.PER_ROW_TIEBREAKING
            )
            if torch.all(argmax_per_row == argmax_per_row[0]):
                per_row_same_selection_count += 1

        # With BATCH_TIEBREAKING, all rows should always select the same index
        self.assertAlmostEqual(batch_same_selection_count / num_trials, 1.0, delta=0.01)

        # With PER_ROW_TIEBREAKING, the probability of all rows selecting the same index
        # should be approximately (1/3)^3 ≈ 0.037 (since each row has 3 tied maximum values)
        expected_probability = (1 / 3) ** 3
        per_row_probability = per_row_same_selection_count / num_trials
        self.assertLess(abs(per_row_probability - expected_probability), 0.05)
