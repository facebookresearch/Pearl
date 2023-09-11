#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (
    NormalDistributionExploration,
)


class TestNormalDistributionExploration(unittest.TestCase):
    """
    This is to test NormalDistributionExploration class
    """

    def test_multi_dimensional_action_scale_with_scalar_max_action(self) -> None:
        """
        Action dimension is set to 3 in this test
        Set noise to 0, so we could test scale specifically
        """
        explore_module = NormalDistributionExploration(
            mean=0,
            std_dev=0,
            max_action_value=3,
            min_action_value=-3,
        )
        action = explore_module.act(exploit_action=torch.tensor([0.5, 0.6, 0.7]))
        self.assertTrue(
            torch.all(torch.isclose(action, torch.tensor([1.5, 1.8, 2.1]), rtol=1e-3))
        )

    def test_multi_dimensional_action_scale_with_vector_max_action(self) -> None:
        """
        Action dimension is set to 3 in this test
        Set noise to 0, so we could test scale specifically
        """
        explore_module = NormalDistributionExploration(
            mean=0,
            std_dev=0,
            max_action_value=torch.tensor([1, 2, 3]),
            min_action_value=torch.tensor([-1, -2, -3]),
        )
        action = explore_module.act(exploit_action=torch.tensor([0.5, 0.6, 0.7]))
        self.assertTrue(
            torch.all(torch.isclose(action, torch.tensor([0.5, 1.2, 2.1]), rtol=1e-3))
        )

    def test_single_dimensional_action_scale_with_scalar_max_action(self) -> None:
        """
        Set noise to 0, so we could test scale specifically
        """
        explore_module = NormalDistributionExploration(
            mean=0, std_dev=0, max_action_value=3, min_action_value=-3
        )
        action = explore_module.act(exploit_action=torch.tensor(0.5))
        self.assertTrue(torch.all(torch.isclose(action, torch.tensor(1.5), rtol=1e-3)))

    def test_multi_dimensional_action_clip_with_scalar_max_action(self) -> None:
        """
        Action dimension is set to 3 in this test
        Set noise to 0, so we could test scale specifically
        """
        explore_module = NormalDistributionExploration(
            mean=0, std_dev=10, max_action_value=3, min_action_value=-3
        )
        action = explore_module.act(exploit_action=torch.tensor([1, 1, 1]))
        self.assertTrue(torch.all(action <= 3))
        action = explore_module.act(exploit_action=torch.tensor([-1, -1, -1]))
        self.assertTrue(torch.all(action >= -3))

    def test_multi_dimensional_action_clip_with_vector_max_action(self) -> None:
        """
        Action dimension is set to 3 in this test
        Set noise to 0, so we could test scale specifically
        """
        max_action_value = torch.tensor([1, 2, 3])
        explore_module = NormalDistributionExploration(
            mean=0,
            std_dev=10,
            max_action_value=max_action_value,
            min_action_value=-max_action_value,
        )
        action = explore_module.act(exploit_action=torch.tensor([1, 1, 1]))
        self.assertTrue(torch.all(action <= max_action_value))
        action = explore_module.act(exploit_action=torch.tensor([-1, -1, -1]))
        self.assertTrue(torch.all(action >= -max_action_value))

    def test_single_dimensional_action_clip_with_scalar_max_action(self) -> None:
        """
        Set noise to 0, so we could test scale specifically
        """
        explore_module = NormalDistributionExploration(
            mean=0, std_dev=10, max_action_value=3, min_action_value=-3
        )
        action = explore_module.act(exploit_action=torch.tensor(1))
        self.assertTrue(torch.all(action <= 3))
        action = explore_module.act(exploit_action=torch.tensor(-1))
        self.assertTrue(torch.all(action >= -3))
