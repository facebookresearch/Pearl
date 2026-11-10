# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import unittest

from pearl.safety_modules.reward_constrained_safety_module import (
    RCSafetyModuleCostCriticContinuousAction,
)
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


class TestRCSafetyModule(unittest.TestCase):
    def setUp(self) -> None:
        self.safety_module = RCSafetyModuleCostCriticContinuousAction(
            constraint_value=0.2,
            state_dim=4,
            action_space=BoxActionSpace(low=-1.0, high=1.0),
            critic_hidden_dims=[8, 8],
        )

    def test_offline_learning_not_supported(self) -> None:
        """Offline safety learning is not supported; learn_batch must raise a clear
        error instead of silently doing nothing."""
        # The batch is never inspected: offline learning is rejected up front.
        with self.assertRaises(AssertionError):
            self.safety_module.learn_batch(None)
