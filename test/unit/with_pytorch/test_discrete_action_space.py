# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
import torch.testing as tt
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDiscreteActionSpace(unittest.TestCase):
    def test_iter(self) -> None:
        # 5 actions with dimention as 4
        actions = [torch.randn(4) for _ in range(5)]
        action_space = DiscreteActionSpace(actions=actions)
        for i, action in enumerate(action_space):
            tt.assert_close(actions[i], action, rtol=0.0, atol=0.0)
