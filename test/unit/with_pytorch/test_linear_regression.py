#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.utils.linear_regression import AvgWeightLinearRegression, LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_linear_regression_random(self) -> None:
        feature_dim = 15
        batch_size = feature_dim * 4  # it is important to have enough data for training

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

        single_test(AvgWeightLinearRegression)
        single_test(LinearRegression)

    def test_state_dict(self) -> None:
        model = LinearRegression(feature_dim=15)
        states = model.state_dict()
        self.assertEqual(len(states), 1)
        self.assertEqual(states["_extra_state"]["A"].shape, (15, 15))
        self.assertEqual(states["_extra_state"]["b"].shape, (15,))
        states["_extra_state"]["b"] = torch.ones((15,))
        model.load_state_dict(states)
        self.assertEqual(model._b[3], 1)
