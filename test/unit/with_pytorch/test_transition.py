# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.replay_buffers.transition import (
    filter_batch_by_bootstrap_mask,
    TransitionBatch,
    TransitionWithBootstrapMaskBatch,
)


class TestTransitionBatch(unittest.TestCase):
    def setUp(self) -> None:
        # Create a simple batch for testing
        self.batch_size = 3
        self.state_dim = 2
        self.action_dim = 1

        self.state = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.action = torch.tensor([[0.1], [0.2], [0.3]])
        self.reward = torch.tensor([[1.0], [2.0], [3.0]])  # Shape: (batch_size, 1)
        self.next_state = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    def test_default_terminated_truncated(self) -> None:
        """Test that terminated and truncated are set to default values when not provided."""
        batch = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
        )

        # Check that terminated is a tensor of True values with shape (batch_size)
        self.assertTrue((batch.terminated == True).all().item())  # noqa: E712
        self.assertEqual(batch.terminated.shape, (self.batch_size,))
        self.assertEqual(batch.terminated.dtype, torch.bool)

        # Check that truncated is a tensor of False values with shape (batch_size)
        self.assertTrue((batch.truncated == False).all().item())  # noqa: E712
        self.assertEqual(batch.truncated.shape, (self.batch_size,))
        self.assertEqual(batch.truncated.dtype, torch.bool)

    def test_custom_terminated_truncated(self) -> None:
        """Test that terminated and truncated use provided values."""
        # Create custom terminated and truncated tensors
        terminated = torch.tensor([True, False, True])
        truncated = torch.tensor([False, True, False])

        batch = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            terminated=terminated,
            truncated=truncated,
        )

        # Check that terminated and truncated use the provided values
        self.assertTrue((batch.terminated == terminated).all().item())
        self.assertTrue((batch.truncated == truncated).all().item())

    def test_device_movement(self) -> None:
        """
        Test that to() method correctly moves terminated
        and truncated to the specified device.
        """
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        batch = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
        )

        # Move batch to CUDA
        cuda_batch = batch.to(torch.device("cuda"))

        # Check that terminated and truncated are moved to CUDA
        self.assertEqual(cuda_batch.terminated.device.type, "cuda")
        self.assertEqual(cuda_batch.truncated.device.type, "cuda")

        # Move batch back to CPU
        cpu_batch = cuda_batch.to(torch.device("cpu"))

        # Check that terminated and truncated are moved back to CPU
        self.assertEqual(cpu_batch.terminated.device.type, "cpu")
        self.assertEqual(cpu_batch.truncated.device.type, "cpu")

    def test_filter_batch_by_bootstrap_mask(self) -> None:
        """Test that filter_batch_by_bootstrap_mask correctly filters terminated and truncated."""
        # Create a batch with bootstrap mask
        bootstrap_mask = torch.tensor(
            [
                [1, 0],  # First transition is active for ensemble 0 only
                [1, 1],  # Second transition is active for both ensembles
                [0, 1],  # Third transition is active for ensemble 1 only
            ]
        )

        batch = TransitionWithBootstrapMaskBatch(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            terminated=torch.tensor([True, False, True]),
            truncated=torch.tensor([False, True, False]),
            bootstrap_mask=bootstrap_mask,
        )

        # Filter batch for ensemble 0
        filtered_batch_0 = filter_batch_by_bootstrap_mask(batch, torch.tensor(0))

        # Check that terminated and truncated are correctly filtered
        self.assertEqual(len(filtered_batch_0), 2)  # Should have 2 transitions
        expected_term = torch.tensor([True, False])
        expected_trunc = torch.tensor([False, True])
        self.assertTrue((filtered_batch_0.terminated == expected_term).all().item())
        self.assertTrue((filtered_batch_0.truncated == expected_trunc).all().item())

        # Filter batch for ensemble 1
        filtered_batch_1 = filter_batch_by_bootstrap_mask(batch, torch.tensor(1))

        # Check that terminated and truncated are correctly filtered
        self.assertEqual(len(filtered_batch_1), 2)  # Should have 2 transitions
        expected_term = torch.tensor([False, True])
        expected_trunc = torch.tensor([True, False])
        self.assertTrue((filtered_batch_1.terminated == expected_term).all().item())
        self.assertTrue((filtered_batch_1.truncated == expected_trunc).all().item())

    def test_mixed_terminated_truncated(self) -> None:
        """Test with mixed terminated and truncated values."""
        # Create a batch with some transitions terminated and some truncated
        terminated = torch.tensor([True, False, False])
        truncated = torch.tensor([False, True, False])

        batch = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            terminated=terminated,
            truncated=truncated,
        )

        # Check individual values
        self.assertTrue(batch.terminated[0].item())
        self.assertFalse(batch.terminated[1].item())
        self.assertFalse(batch.terminated[2].item())

        self.assertFalse(batch.truncated[0].item())
        self.assertTrue(batch.truncated[1].item())
        self.assertFalse(batch.truncated[2].item())

    def test_none_terminated_truncated(self) -> None:
        """Test with None terminated and truncated values."""
        # Create a batch with None terminated and truncated
        batch = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            # default terminated and truncated
        )

        # Check that terminated is a tensor of True values
        self.assertTrue((batch.terminated == True).all().item())  # noqa: E712
        self.assertEqual(batch.terminated.shape, (self.batch_size,))

        # Check that truncated is a tensor of False values
        self.assertTrue((batch.truncated == False).all().item())  # noqa: E712
        self.assertEqual(batch.truncated.shape, (self.batch_size,))

    def test_reward_and_action_shapes(self) -> None:
        """Test that reward and action can have different shapes."""
        # Test with reward shape (batch_size,)
        reward_1d = torch.tensor([1.0, 2.0, 3.0])  # Shape: (batch_size,)

        batch_1d = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=reward_1d,
            next_state=self.next_state,
        )

        # Verify that the batch was created successfully
        self.assertEqual(batch_1d.reward.shape, (self.batch_size,))

        # Test with reward shape (batch_size, 1)
        reward_2d = torch.tensor([[1.0], [2.0], [3.0]])  # Shape: (batch_size, 1)

        batch_2d = TransitionBatch(
            state=self.state,
            action=self.action,
            reward=reward_2d,
            next_state=self.next_state,
        )

        # Verify that the batch was created successfully
        self.assertEqual(batch_2d.reward.shape, (self.batch_size, 1))

        # Test with action shape (batch_size,)
        action_1d = torch.tensor([0.1, 0.2, 0.3])  # Shape: (batch_size,)

        batch_action_1d = TransitionBatch(
            state=self.state,
            action=action_1d,
            reward=self.reward,
            next_state=self.next_state,
        )

        # Verify that the batch was created successfully
        self.assertEqual(batch_action_1d.action.shape, (self.batch_size,))

        # Test with action shape (batch_size, action_dim)
        action_2d = torch.tensor(
            [[0.1], [0.2], [0.3]]
        )  # Shape: (batch_size, action_dim)

        batch_action_2d = TransitionBatch(
            state=self.state,
            action=action_2d,
            reward=self.reward,
            next_state=self.next_state,
        )

        # Verify that the batch was created successfully
        self.assertEqual(
            batch_action_2d.action.shape, (self.batch_size, self.action_dim)
        )

    def test_multidimensional_tensors(self) -> None:
        """Test that tensors with more than 2 dimensions are accepted."""
        # Create tensors with 3 dimensions
        batch_size = 3
        state_dim = 2
        action_dim = 1
        extra_dim = 4

        # Shape: (batch_size, state_dim, extra_dim)
        state = torch.rand(batch_size, state_dim, extra_dim)
        # Test with action shape (batch_size,)
        action_1d = torch.rand(batch_size)  # Shape: (batch_size,)
        # Test with both reward shapes
        reward_1d = torch.tensor([1.0, 2.0, 3.0])  # Shape: (batch_size,)
        # Shape: (batch_size, state_dim, extra_dim)
        next_state = torch.rand(batch_size, state_dim, extra_dim)

        # Create batch with multidimensional state, 1D action, and 1D reward
        batch_1d = TransitionBatch(
            state=state,
            action=action_1d,
            reward=reward_1d,
            next_state=next_state,
        )

        # Verify that the batch was created successfully
        self.assertEqual(batch_1d.state.shape, (batch_size, state_dim, extra_dim))
        self.assertEqual(batch_1d.action.shape, (batch_size,))
        self.assertIsNotNone(batch_1d.next_state)
        if batch_1d.next_state is not None:  # Type narrowing for mypy
            self.assertEqual(
                batch_1d.next_state.shape, (batch_size, state_dim, extra_dim)
            )

        # Test with action shape (batch_size, action_dim, extra_dim)
        action_3d = torch.rand(batch_size, action_dim, extra_dim)

        # Create batch with multidimensional tensors and 1D reward
        batch_3d = TransitionBatch(
            state=state,
            action=action_3d,
            reward=reward_1d,
            next_state=next_state,
        )

        # Verify that the batch was created successfully
        self.assertEqual(batch_3d.state.shape, (batch_size, state_dim, extra_dim))
        self.assertEqual(batch_3d.action.shape, (batch_size, action_dim, extra_dim))
        self.assertIsNotNone(batch_3d.next_state)
        if batch_3d.next_state is not None:  # Type narrowing for mypy
            self.assertEqual(
                batch_3d.next_state.shape, (batch_size, state_dim, extra_dim)
            )

        # Test with even more dimensions and 2D reward
        state_4d = torch.rand(batch_size, state_dim, extra_dim, 2)
        action_4d = torch.rand(batch_size, action_dim, extra_dim, 2)
        reward_2d = torch.tensor([[1.0], [2.0], [3.0]])  # Shape: (batch_size, 1)
        next_state_4d = torch.rand(batch_size, state_dim, extra_dim, 2)

        batch_4d = TransitionBatch(
            state=state_4d,
            action=action_4d,
            reward=reward_2d,
            next_state=next_state_4d,
        )

        # Verify that the batch was created successfully
        self.assertEqual(batch_4d.state.shape, (batch_size, state_dim, extra_dim, 2))
        self.assertEqual(batch_4d.action.shape, (batch_size, action_dim, extra_dim, 2))
        self.assertIsNotNone(batch_4d.next_state)
        if batch_4d.next_state is not None:  # Type narrowing for mypy
            self.assertEqual(
                batch_4d.next_state.shape, (batch_size, state_dim, extra_dim, 2)
            )
