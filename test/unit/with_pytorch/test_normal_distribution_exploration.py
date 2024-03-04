# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from gym.spaces import Box  # noqa

from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (
    NormalDistributionExploration,
)
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


class TestNormalDistributionExploration(unittest.TestCase):
    """
    This is to test NormalDistributionExploration class.
    """

    def test_multi_dimensional_action_space(self) -> None:
        low, high = torch.tensor([-4, -0.4, -0.04]), torch.tensor([4, 0.4, 0.04])
        action_space = BoxActionSpace(low=low, high=high)
        exploration_module = NormalDistributionExploration(
            mean=0,
            std_dev=1,
        )
        exploit_action = torch.tensor([0.5, 0.3, 0.01])
        action = exploration_module.act(
            exploit_action=exploit_action, action_space=action_space
        )
        self.assertTrue(torch.all(action <= high))
        self.assertTrue(torch.all(action >= low))
