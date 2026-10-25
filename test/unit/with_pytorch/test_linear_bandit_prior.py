# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import numpy as np
import torch
import torch.testing as tt
from parameterized import parameterized
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (  # noqa: E501
    ThompsonSamplingExplorationLinear,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestLinearBanditPrior(unittest.TestCase):
    @parameterized.expand(
        [(regularization, pinv) for regularization in (0.25, 1.0, 4.0) for pinv in (False, True)]
    )
    def test_prior_uncertainty(self, regularization: float, pinv: bool) -> None:
        model = LinearRegression(2, l2_reg_lambda=regularization, force_pinv=pinv)
        features = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        # With no data, A = lambda * I, including the intercept feature.
        expected = torch.tensor([[1.0], [math.sqrt(26.0)]]) / math.sqrt(regularization)
        tt.assert_close(model.calculate_sigma(features), expected)
        tt.assert_close(model.calculate_sigma(features.unsqueeze(0)), expected.reshape(1, 2))
        tt.assert_close(model(features), torch.zeros(2, 1))

    def test_zero_weight_batch_preserves_uncertainty(self) -> None:
        model = LinearRegression(2)
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        before = model.calculate_sigma(features)
        model.learn_batch(features, torch.zeros(2), weight=torch.zeros(2))
        tt.assert_close(model.calculate_sigma(features), before)

    def test_initial_coefficients_and_checkpoint(self) -> None:
        coefficients = torch.tensor([1.0, 2.0, 3.0])
        model = LinearRegression(2, initial_coefs=coefficients).double()
        features = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        tt.assert_close(model(features), torch.tensor([[19.0]], dtype=torch.float64))
        tt.assert_close(model.coefs, coefficients.double())
        tt.assert_close(
            model.calculate_sigma(features),
            torch.tensor([[math.sqrt(26.0)]], dtype=torch.float64),
        )
        model.learn_batch(features, torch.tensor([5.0], dtype=torch.float64))
        restored = LinearRegression(2).double()
        restored.load_state_dict(model.state_dict())
        tt.assert_close(restored(features), model(features))
        tt.assert_close(restored.calculate_sigma(features), model.calculate_sigma(features))

    def test_first_update_matches_weighted_ridge_reference(self) -> None:
        model = LinearRegression(2)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        rewards = torch.tensor([2.0, -1.0])
        weights = torch.tensor([2.0, 0.5])
        model.learn_batch(features, rewards, weight=weights)
        augmented = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        inverse = np.linalg.inv(np.eye(3) + augmented.T @ np.diag([2.0, 0.5]) @ augmented)
        reference_coefs = inverse @ augmented.T @ np.diag([2.0, 0.5]) @ np.array([2.0, -1.0])
        expected_sigma = np.sqrt(np.einsum("bi,ij,bj->b", augmented, inverse, augmented))
        tt.assert_close(model.coefs, torch.tensor(reference_coefs, dtype=torch.float32))
        tt.assert_close(
            model.calculate_sigma(features).flatten(),
            torch.tensor(expected_sigma, dtype=torch.float32),
        )

    def test_unregularized_initial_state(self) -> None:
        model = LinearRegression(2, l2_reg_lambda=0, force_pinv=True)
        features = torch.tensor([[1.0, 2.0]])
        tt.assert_close(model.calculate_sigma(features), torch.zeros(1, 1))
        model.learn_batch(features, torch.zeros(1), weight=torch.zeros(1))
        tt.assert_close(model.calculate_sigma(features), torch.zeros(1, 1))

    def test_cold_start_ucb_scores_and_action(self) -> None:
        bandit = LinearBandit(2, exploration_module=UCBExploration(alpha=1))
        actions = DiscreteActionSpace([torch.tensor([0.0]), torch.tensor([2.0])])
        state = torch.tensor([0.0])
        tt.assert_close(bandit.get_scores(state, actions), torch.tensor([[1.0, math.sqrt(5.0)]]))
        tt.assert_close(bandit.act(state, actions), torch.tensor([[2.0]]))

    def test_efficient_thompson_prior_variance(self) -> None:
        model = LinearRegression(1)
        exploration = ThompsonSamplingExplorationLinear(enable_efficient_sampling=True)
        actions = DiscreteActionSpace([torch.tensor([0.0]), torch.tensor([2.0])])
        features = torch.tensor([[[0.0], [2.0]]]).expand(4096, -1, -1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            scores = exploration.get_scores(
                features, actions, model(features), representation=model
            )
        # Check sampling behavior against analytic marginal prior variances.
        tt.assert_close(scores.var(dim=0), torch.tensor([1.0, 5.0]), rtol=0.1, atol=0.0)


if __name__ == "__main__":
    unittest.main()
