#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

from pearl.core.sequential_decision_making.policy_learners.ppo import (
    ProximalPolicyOptimization,
)
from pearl.utils.action_spaces import DiscreteActionSpace


class TestPPO(unittest.TestCase):
    def test_optimizer_param_count(self):
        """
        This test is to ensure optimizer defined in PPO has all the parameters needed
        including actor and critic
        """
        policy_learner = ProximalPolicyOptimization(
            16,
            DiscreteActionSpace(range(3)),
            [64, 64],
            training_rounds=1,
            batch_size=500,
            epsilon=0.1,
        )
        optimizer_params_count = sum(
            len(group["params"]) for group in policy_learner._optimizer.param_groups
        )
        model_params_count = sum([1 for _ in policy_learner._actor.parameters()]) + sum(
            [1 for _ in policy_learner._critic.parameters()]
        )
        self.assertEqual(optimizer_params_count, model_params_count)
