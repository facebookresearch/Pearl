# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class PropensityMaskRegressionTest(unittest.TestCase):
    def test_unavailable_actions_are_not_sampled(self) -> None:
        space = DiscreteActionSpace([torch.tensor([0.0]), torch.tensor([1.0])])
        exploration = PropensityExploration()
        for mask in [torch.tensor([False, True]), torch.tensor([0.0, 1.0])]:
            with self.subTest(dtype=mask.dtype):
                values = torch.tensor([0.9, 0.1])
                original = values.clone()
                torch.manual_seed(7)
                for _ in range(20):
                    action = exploration.act(
                        torch.zeros(1), space, values, action_availability_mask=mask
                    )
                    torch.testing.assert_close(action, torch.tensor([1.0]))
                torch.testing.assert_close(values, original)

    def test_all_available_mask_preserves_seeded_sampling(self) -> None:
        space = DiscreteActionSpace([torch.tensor([0.0]), torch.tensor([1.0])])
        exploration = PropensityExploration()
        values = torch.tensor([0.3, 0.7])
        torch.manual_seed(17)
        expected = [exploration.act(torch.zeros(1), space, values) for _ in range(20)]
        torch.manual_seed(17)
        actual = [
            exploration.act(torch.zeros(1), space, values, action_availability_mask=torch.ones(2))
            for _ in range(20)
        ]
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
