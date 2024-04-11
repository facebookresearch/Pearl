# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.utils.functional_utils.learning.action_utils import (
    argmax_random_tie_breaks,
    get_model_action_index_batch,
)


class TestGetAction(unittest.TestCase):
    def test_argmax_random_tie_breaks_no_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks(scores)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(argmax_values_returned[0], {1, 2})
        self.assertSetEqual(argmax_values_returned[1], {0, 1})
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(argmax_values_returned[3], {1, 2})

    def test_argmax_random_tie_breaks_mask(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = argmax_random_tie_breaks(scores, mask)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_randomize(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_action_index_batch(scores, mask, randomize_ties=True)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(argmax_values_returned[2], {0, 2})
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )

    def test_get_model_actions_not_randomize(self) -> None:
        scores = torch.tensor(
            [[1, 20, 20], [4, 4, 3], [15, 10, 15], [100, float("inf"), float("inf")]]
        )
        mask = torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]])
        argmax_values_returned = {0: set(), 1: set(), 2: set(), 3: set()}
        for _ in range(1000):
            # repeat many times since the function is stochastic
            argmax = get_model_action_index_batch(scores, mask, randomize_ties=False)
            # make sure argmax returns one of the max element indices
            argmax_values_returned[0].add(argmax[0].item())
            argmax_values_returned[1].add(argmax[1].item())
            argmax_values_returned[2].add(argmax[2].item())
            argmax_values_returned[3].add(argmax[3].item())
        self.assertSetEqual(
            argmax_values_returned[0],
            {
                1,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[1],
            {
                2,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[2],
            {
                0,
            },
        )
        self.assertSetEqual(
            argmax_values_returned[3],
            {
                2,
            },
        )
