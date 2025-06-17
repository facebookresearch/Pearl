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
from pearl.utils.functional_utils.learning.action_utils import (
    argmax_random_tie_breaks,
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


class TestGetAction(unittest.TestCase):
    def test_argmax_random_tie_breaks_no_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks(scores)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(argmax_values_returned[0], {1, 2})
        self.assertSetEqual(argmax_values_returned[1], {0, 1})
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(argmax_values_returned[3], {1, 2})

    def test_argmax_random_tie_breaks_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks(scores, mask)
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
            argmax = get_model_action_index_batch(scores, mask, randomize_ties=True)
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
            argmax = get_model_action_index_batch(scores, mask, randomize_ties=False)
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
