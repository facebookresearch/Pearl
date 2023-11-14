#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from gym.spaces import Box  # noqa

from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (
    NormalDistributionExploration,
)


class TestNormalDistributionExploration(unittest.TestCase):
    """
    This is to test NormalDistributionExploration class.
    """

    # TODO: add for asymmetric action bounds
    def test_multi_dimensional_action_space(self) -> None:
        low, high = -4, 4
        action_space = Box(low=low, high=high, shape=(3,))
        exploration_module = NormalDistributionExploration(
            mean=0,
            std_dev=1,
        )
        exploit_action = torch.tensor([0.5, 0.6, 0.7])
        # TODO: type of action_space should not be Box but ActionSpace
        action = exploration_module.act(
            exploit_action=exploit_action, action_space=action_space  # pyre-ignore
        )
        self.assertTrue(torch.all(action <= high))
        self.assertTrue(torch.all(action >= low))
