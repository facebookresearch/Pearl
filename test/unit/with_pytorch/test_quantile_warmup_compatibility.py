# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.quantile_regression_deep_q_learning import (
    QuantileRegressionDeepQLearning,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestNativeQuantileWarmup(unittest.TestCase):
    def test_native_warmup_returns_catalog_actions_before_and_after_warmup(self):
        for learner_type in (DeepQLearning, QuantileRegressionDeepQLearning):
            with self.subTest(learner=learner_type.__name__):
                torch.manual_seed(17)
                catalog = DiscreteActionSpace([torch.tensor([0]), torch.tensor([1])])
                exploration = Warmup(EGreedyExploration(0.0), warmup_steps=2)
                learner = learner_type(
                    state_dim=2,
                    action_space=catalog,
                    hidden_dims=[8],
                    exploration_module=exploration,
                )
                agent = PearlAgent(policy_learner=learner)
                agent.reset(torch.tensor([0.5, 1.0]), catalog)
                for step in range(4):
                    action = agent.act()
                    self.assertEqual(
                        action.numel(), 1, f"non-action returned at step {step}"
                    )
                    self.assertIn(int(action.item()), (0, 1))
                count = exploration.time_step
                greedy = agent.act(exploit=True)
                self.assertEqual(greedy.numel(), 1)
                self.assertEqual(exploration.time_step, count)
