# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class DiscreteMaskRegressionTest(unittest.TestCase):
    def test_available_action_masks_use_gymnasium_dtype(self) -> None:
        for cls in [DiscreteSpace, DiscreteActionSpace]:
            for dtype in [torch.bool, torch.int32, torch.int64, torch.float32]:
                with self.subTest(space=cls.__name__, dtype=dtype):
                    space = cls([torch.tensor([10]), torch.tensor([20])])
                    mask = torch.tensor([0, 1], dtype=dtype)
                    original = mask.clone()
                    torch.testing.assert_close(space.sample(mask), torch.tensor([20]))
                    torch.testing.assert_close(mask, original)

    def test_empty_mask_keeps_first_element_fallback(self) -> None:
        space = DiscreteActionSpace([torch.tensor([10]), torch.tensor([20])])
        torch.testing.assert_close(space.sample(torch.tensor([0, 0])), torch.tensor([10]))

    def test_unmasked_sampling_stays_available(self) -> None:
        space = DiscreteActionSpace([torch.tensor([10]), torch.tensor([20])], seed=7)
        for _ in range(10):
            self.assertIn(space.sample().item(), [10, 20])
