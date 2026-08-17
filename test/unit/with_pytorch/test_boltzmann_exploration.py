# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import unittest

import torch
from pearl.policy_learners.exploration_modules.common.boltzmann_exploration import (
    BoltzmannExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestBoltzmannExploration(unittest.TestCase):
    """Tests for BoltzmannExploration (softmax(Q/T) score exploration)."""

    def test_invalid_temperature_raises(self) -> None:
        with self.assertRaises(ValueError):
            BoltzmannExploration(temperature=0.0)
        with self.assertRaises(ValueError):
            BoltzmannExploration(temperature=-1.0)

    def test_get_scores_returns_valid_probability_distribution(self) -> None:
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0])]
        )
        exploration = BoltzmannExploration(temperature=1.0)
        values = torch.tensor([[1.0, 2.0, 0.5]])
        scores = exploration.get_scores(
            subjective_state=torch.zeros(1, 4),
            action_space=action_space,
            values=values,
        )
        self.assertEqual(scores.shape, (1, 3))
        self.assertTrue(torch.all(scores >= 0.0))
        self.assertTrue(torch.all(scores <= 1.0))
        self.assertTrue(torch.allclose(scores.sum(dim=-1), torch.ones(1)))

    def test_get_scores_matches_torch_softmax(self) -> None:
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([0.0]), torch.tensor([1.0])]
        )
        exploration = BoltzmannExploration(temperature=2.0)
        values = torch.tensor([[3.0, -1.0]])
        scores = exploration.get_scores(
            subjective_state=torch.zeros(1, 4),
            action_space=action_space,
            values=values,
        )
        expected = torch.softmax(values / 2.0, dim=-1)
        self.assertTrue(torch.allclose(scores, expected))

    def test_temperature_controls_sharpness(self) -> None:
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([0.0]), torch.tensor([1.0])]
        )
        values = torch.tensor([[2.0, 0.0]])

        sharp = BoltzmannExploration(temperature=0.1)
        flat = BoltzmannExploration(temperature=10.0)

        sharp_scores = sharp.get_scores(
            subjective_state=torch.zeros(1, 4),
            action_space=action_space,
            values=values,
        )
        flat_scores = flat.get_scores(
            subjective_state=torch.zeros(1, 4),
            action_space=action_space,
            values=values,
        )
        # Lower temperature => sharper distribution => higher prob on argmax.
        self.assertGreater(sharp_scores[0, 0].item(), flat_scores[0, 0].item())

    def test_compare_detects_temperature_difference(self) -> None:
        a = BoltzmannExploration(temperature=1.0)
        b = BoltzmannExploration(temperature=2.0)
        diff = a.compare(b)
        self.assertIn("temperature is different", diff)

    def test_compare_detects_other_type(self) -> None:
        from pearl.policy_learners.exploration_modules.common.no_exploration import (
            NoExploration,
        )

        a = BoltzmannExploration(temperature=1.0)
        diff = a.compare(NoExploration())
        self.assertIn("not an instance of BoltzmannExploration", diff)

    def test_compare_equal_instances_returns_no_temperature_diff(self) -> None:
        a = BoltzmannExploration(temperature=1.5)
        b = BoltzmannExploration(temperature=1.5)
        diff = a.compare(b)
        # When temperatures match and types match, no specific diff lines from
        # BoltzmannExploration should be appended.
        self.assertNotIn("temperature is different", diff)
        self.assertNotIn("not an instance of BoltzmannExploration", diff)

    def test_act_samples_stochastically(self) -> None:
        """`act` must sample from `softmax(Q / T)` (not argmax). With low
        temperature on near-tied Q-values, sampling will still occasionally
        pick the minority action -- argmax never would.
        """
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([0.0]), torch.tensor([1.0])]
        )
        # Near-tied Q-values with mild preference for action 0.
        # At T=1.0, softmax([1.0, 0.7]) ≈ [0.574, 0.426] -- a greedy policy
        # would always pick 0. A stochastic policy picks 1 ~42% of the time.
        exploration = BoltzmannExploration(temperature=1.0)
        values = torch.tensor([[1.0, 0.7]] * 1000)  # 1000 identical samples
        torch.manual_seed(0)
        actions = exploration.act(
            subjective_state=torch.zeros(1000, 4),
            action_space=action_space,
            values=values,
        )
        # Count how many of the 1000 samples picked action 1.
        # Greedy would yield 0 of these. Stochastic should yield ~426.
        n_action_1 = int((actions.flatten() == 1.0).sum().item())
        self.assertGreater(
            n_action_1,
            100,
            f"Expected ~426 action-1 samples (softmax([1.0, 0.7]) ≈ [0.57, 0.43]); "
            f"got {n_action_1}. If 0, the act() override is missing (still greedy).",
        )
        self.assertLess(n_action_1, 700)

    def test_act_respects_availability_mask(self) -> None:
        """Masked actions must never be sampled, even if their Q-value is high."""
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([0.0]), torch.tensor([1.0])]
        )
        exploration = BoltzmannExploration(temperature=1.0)
        values = torch.tensor([[5.0, 0.0]] * 200)  # strongly prefer action 0
        # Mask out action 0; only action 1 should ever be sampled.
        mask = torch.tensor([[0.0, 1.0]] * 200)
        torch.manual_seed(0)
        actions = exploration.act(
            subjective_state=torch.zeros(200, 4),
            action_space=action_space,
            values=values,
            action_availability_mask=mask,
        )
        # All sampled actions should be action 1 (value 1.0).
        self.assertTrue(torch.all(actions.flatten() == 1.0))
