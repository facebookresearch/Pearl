# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import (
    FastCBExploration,
    SquareCBExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestSquareCBExploration(unittest.TestCase):
    """Tests for SquareCBExploration.act over batched value tensors."""

    def test_act_batched_states_does_not_crash(self) -> None:
        # batch_size (2) != action_count (3) exercises the gap broadcasting:
        # ``empirical_gaps = max_val.unsqueeze(1) - values`` must align the
        # per-row max with the (batch_size, action_count) values.
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([i]) for i in range(3)]
        )
        exploration = SquareCBExploration(gamma=10.0)
        values = torch.tensor([[0.10, 0.20, 0.90], [0.80, 0.30, 0.10]])
        torch.manual_seed(0)
        actions = exploration.act(
            subjective_state=torch.zeros(2, 4),
            action_space=action_space,
            values=values,
        )
        self.assertEqual(actions.shape[0], 2)
        self.assertTrue(int(actions.min()) >= 0)
        self.assertTrue(int(actions.max()) < action_space.n)

    def test_act_probabilities_are_per_row_valid_distributions(self) -> None:
        # The greedy action's residual probability must be computed from the
        # current row only (sum over that row), so every row of the sampling
        # distribution sums to 1 and the greedy action carries the most mass.
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([i]) for i in range(3)]
        )
        exploration = SquareCBExploration(gamma=10.0)
        values = torch.tensor([[0.10, 0.20, 0.90], [0.80, 0.30, 0.10]])

        # Reconstruct, per row, the distribution act() builds via the module's
        # own get_unnormalize_prob (no randomness involved).
        max_val, max_indices = torch.max(values, dim=1)
        empirical_gaps = max_val.unsqueeze(1) - values
        rows = []
        for b in range(values.size(0)):
            prob = exploration.get_unnormalize_prob(
                empirical_gaps[b, :], max_val[b], action_space.n
            )
            prob[max_indices[b]] = 0.0
            prob[max_indices[b]] = 1.0 - torch.sum(prob)
            rows.append(prob)
        prob = torch.stack(rows)

        self.assertTrue(torch.allclose(prob.sum(dim=1), torch.ones(2), atol=1e-6))
        # Greedy action (argmax of values) should be the most probable per row.
        self.assertTrue(torch.equal(prob.argmax(dim=1), values.argmax(dim=1)))

    def test_act_single_state(self) -> None:
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([i]) for i in range(3)]
        )
        exploration = SquareCBExploration(gamma=10.0)
        torch.manual_seed(0)
        action = exploration.act(
            subjective_state=torch.zeros(1, 4),
            action_space=action_space,
            values=torch.tensor([[0.10, 0.20, 0.90]]),
        )
        self.assertTrue(0 <= int(action) < action_space.n)

    def test_fastcb_act_batched_states(self) -> None:
        # FastCBExploration inherits act() and overrides get_unnormalize_prob
        # with a branch on max_val; act() must therefore feed it a scalar row
        # maximum so batched input does not raise on an ambiguous truth value.
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([i]) for i in range(3)]
        )
        exploration = FastCBExploration(gamma=10.0)
        values = torch.tensor([[0.10, 0.20, 0.90], [0.80, 0.30, 0.10]])
        torch.manual_seed(0)
        actions = exploration.act(
            subjective_state=torch.zeros(2, 4),
            action_space=action_space,
            values=values,
        )
        self.assertEqual(actions.shape[0], 2)
        self.assertTrue(int(actions.min()) >= 0)
        self.assertTrue(int(actions.max()) < action_space.n)
