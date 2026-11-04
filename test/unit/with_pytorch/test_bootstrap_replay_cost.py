# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import (
    BootstrapReplayBuffer,
)


class BootstrapCostRegressionTest(unittest.TestCase):
    def test_sample_preserves_costs_and_bootstrap_masks(self) -> None:
        buffer = BootstrapReplayBuffer(10, p=0.5, ensemble_size=3)
        buffer.is_action_continuous = True
        for index, cost in enumerate([0.0, 0.25, 0.75]):
            buffer.push(
                torch.tensor([float(index)]),
                torch.tensor([0.0]),
                1.0,
                False,
                False,
                next_state=torch.tensor([float(index + 1)]),
                cost=cost,
            )
        batch = buffer.sample(3)
        self.assertIsNotNone(batch.cost)
        torch.testing.assert_close(batch.cost.sort().values, torch.tensor([0.0, 0.25, 0.75]))
        self.assertEqual(batch.bootstrap_mask.shape, (3, 3))
        for index in range(3):
            source_index = int(batch.state[index, 0].item())
            torch.testing.assert_close(
                batch.bootstrap_mask[index], buffer.memory[source_index].bootstrap_mask[0]
            )

    def test_missing_cost_remains_optional(self) -> None:
        buffer = BootstrapReplayBuffer(10, p=1.0, ensemble_size=2)
        buffer.is_action_continuous = True
        buffer.push(torch.tensor([0.0]), torch.tensor([0.0]), 1.0, True, False)
        batch = buffer.sample(1)
        self.assertIsNone(batch.cost)
        torch.testing.assert_close(batch.bootstrap_mask, torch.ones(1, 2))
