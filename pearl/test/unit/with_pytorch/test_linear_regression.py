#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.utils.functional_utils.learning.linear_regression import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_append_ones(self) -> None:
        # 1D input
        x = torch.randn(10)
        expected_output = torch.cat([torch.ones((1,)), x], dim=0)
        output = LinearRegression.append_ones(x)
        self.assertTrue(torch.allclose(expected_output, output))

        # 2D input
        x = torch.randn(10, 5)
        expected_output = torch.cat([torch.ones((10, 1)), x], dim=1)
        output = LinearRegression.append_ones(x)
        self.assertTrue(torch.allclose(expected_output, output))

        # 3D input
        x = torch.randn(10, 5, 6)
        expected_output = torch.cat([torch.ones((10, 5, 1)), x], dim=2)
        output = LinearRegression.append_ones(x)
        self.assertTrue(torch.allclose(expected_output, output))

        # make sure it's traceable
        _ = torch.fx.symbolic_trace(LinearRegression.append_ones)

    def test_linear_regression_random(self) -> None:
        feature_dim = 15
        batch_size = feature_dim * 4  # it is important to have enough data for training

        # pyre-fixme[53]: Captured variable `batch_size` is not annotated.
        # pyre-fixme[53]: Captured variable `feature_dim` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def single_test(linear_regression_class):
            linear_regression = linear_regression_class(feature_dim=feature_dim)
            losses = []
            feature = torch.randn(batch_size, feature_dim)
            reward = feature.sum(-1)
            weight = torch.ones(batch_size)
            for _ in range(10):
                loss = (linear_regression(feature) - reward) ** 2
                losses.append(loss.mean().item())
                linear_regression.learn_batch(x=feature, y=reward, weight=weight)
            losses.append(loss.mean().item())

            self.assertGreater(sum(losses[:5]), sum(losses[-5:]))
            self.assertGreater(1e-2, losses[-1])

        single_test(LinearRegression)

    def test_state_dict(self) -> None:
        feature_dim = 15
        model = LinearRegression(feature_dim=feature_dim)
        states = model.state_dict()
        self.assertEqual(len(states), 5)
        self.assertEqual(states["_A"].shape, (feature_dim + 1, feature_dim + 1))
        self.assertEqual(states["_b"].shape, (feature_dim + 1,))
        self.assertEqual(states["_inv_A"].shape, (feature_dim + 1, feature_dim + 1))
        self.assertEqual(states["_coefs"].shape, (feature_dim + 1,))
        self.assertEqual(states["_sum_weight"].shape, (1,))
        states["_b"] = torch.ones((feature_dim + 1,))
        model.load_state_dict(states)
        self.assertEqual(model._b[3], 1)
