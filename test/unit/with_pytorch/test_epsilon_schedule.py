# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class EpsilonScheduleRegressionTest(unittest.TestCase):
    def test_schedule_reaches_and_keeps_end_epsilon(self) -> None:
        space = DiscreteActionSpace(actions=[torch.tensor([0.0])])
        for start, end in [(1.0, 0.0), (0.0, 1.0)]:
            with self.subTest(start=start, end=end):
                exploration = EGreedyExploration(0.5, start, end, 2)
                observed = []
                for _ in range(5):
                    exploration.act(torch.zeros(1), space, torch.tensor([0.0]))
                    observed.append(exploration.curr_epsilon)
                self.assertEqual(observed, [start, 0.5, end, end, end])

    def test_zero_warmup_uses_end_immediately(self) -> None:
        exploration = EGreedyExploration(0.5, 1.0, 0.0, 0)
        space = DiscreteActionSpace(actions=[torch.tensor([0.0])])
        exploration.act(torch.zeros(1), space, torch.tensor([0.0]))
        self.assertEqual(exploration.curr_epsilon, 0.0)

    def test_constant_epsilon_remains_unchanged(self) -> None:
        exploration = EGreedyExploration(0.25)
        space = DiscreteActionSpace(actions=[torch.tensor([0.0])])
        for _ in range(5):
            exploration.act(torch.zeros(1), space, torch.tensor([0.0]))
            self.assertEqual(exploration.curr_epsilon, 0.25)
