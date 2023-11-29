#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDiscreteActionSpace(unittest.TestCase):
    def test_iter(self) -> None:
        # 5 actions with dimention as 4
        actions = [torch.randn(4) for _ in range(5)]
        action_space = DiscreteActionSpace(actions=actions)
        for i, action in enumerate(action_space):
            self.assertTrue(torch.equal(actions[i], action))
