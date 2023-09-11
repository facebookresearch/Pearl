#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import random
import unittest

from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestDiscreteActionSpace(unittest.TestCase):
    def test_iter(self) -> None:
        # 5 actions with dimention as 4
        actions = [[random.random() for _ in range(4)] for _ in range(5)]
        action_space = DiscreteActionSpace(actions=actions)
        for i, action in enumerate(action_space):
            self.assertEqual(actions[i], action)
