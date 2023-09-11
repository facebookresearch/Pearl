#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestDiscreteActionSpace(unittest.TestCase):
    def test_cat_state_tensor(self) -> None:
        # action_count = 3
        action_space = DiscreteActionSpace([[1, 2], [3, 4], [5, 6]])
        # batch_size = 1 feature_size = 1+2 = 3
        new_tensor = action_space.cat_state_tensor(
            torch.Tensor([1])
        )  # (batch_size, action_count, feature_size)
        self.assertEqual(new_tensor.tolist(), [[[1, 1, 2], [1, 3, 4], [1, 5, 6]]])
        # batch_size = 2 feature_size = 1+2 = 3
        new_tensor = action_space.cat_state_tensor(
            torch.Tensor([[1], [2]])
        )  # (batch_size, action_count, feature_size)
        self.assertEqual(
            new_tensor.tolist(),
            [[[1, 1, 2], [1, 3, 4], [1, 5, 6]], [[2, 1, 2], [2, 3, 4], [2, 5, 6]]],
        )
