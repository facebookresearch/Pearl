# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import unittest

import torch
from pearl.neural_networks.common.noisy_linear import NoisyLinear
from pearl.neural_networks.sequential_decision_making.rainbow_q_value_network import (
    RainbowQValueNetwork,
)
from torch.nn import functional as F


class TestNoisyLinear(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_factorized_noise_matches_two_gaussian_draws(self) -> None:
        layer = NoisyLinear(3, 2)
        torch.manual_seed(29)
        input_draw = torch.randn(3)
        output_draw = torch.randn(2)
        input_noise = input_draw.sign() * input_draw.abs().sqrt()
        output_noise = output_draw.sign() * output_draw.abs().sqrt()
        torch.manual_seed(29)
        layer.reset_noise()
        torch.testing.assert_close(
            layer.weight_epsilon, output_noise.outer(input_noise)
        )
        torch.testing.assert_close(layer.bias_epsilon, output_noise)

    def test_noise_is_fixed_until_reset_and_disabled_for_evaluation(self) -> None:
        layer = NoisyLinear(3, 2)
        inputs = torch.ones(4, 3)
        first = layer(inputs)
        torch.testing.assert_close(first, layer(inputs), rtol=0, atol=0)
        layer.reset_noise()
        self.assertFalse(torch.equal(first, layer(inputs)))

        expected = F.linear(inputs, layer.weight_mu, layer.bias_mu)
        layer.eval()
        torch.testing.assert_close(layer(inputs), expected)
        layer.reset_noise()
        torch.testing.assert_close(layer(inputs), expected)
        layer.train()
        layer.enable_noise(False)
        torch.testing.assert_close(layer(inputs), expected)
        layer.enable_noise(True)
        self.assertFalse(torch.equal(layer(inputs), expected))

    def test_noise_scale_and_mean_receive_gradients(self) -> None:
        layer = NoisyLinear(3, 2)
        layer(torch.ones(2, 3)).square().sum().backward()
        for parameter in (
            layer.weight_mu,
            layer.weight_sigma,
            layer.bias_mu,
            layer.bias_sigma,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_no_bias_and_float64_noise(self) -> None:
        layer = NoisyLinear(3, 2, bias=False).double()
        layer.reset_noise()
        self.assertIsNone(layer.bias_mu)
        self.assertIsNone(layer.bias_epsilon)
        self.assertEqual(layer.weight_epsilon.dtype, torch.float64)
        self.assertEqual(layer(torch.ones(4, 3, dtype=torch.float64)).shape, (4, 2))

    def test_invalid_dimensions_and_noise_scale(self) -> None:
        for args in ((0, 2, 0.5), (2, 0, 0.5), (2, 2, -1), (2, 2, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                NoisyLinear(args[0], args[1], std_init=args[2])


class TestRainbowQValueNetwork(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)

    def _network(self, **kwargs: object) -> RainbowQValueNetwork:
        options = dict(
            state_dim=2,
            action_dim=1,
            hidden_dims=[],
            num_quantiles=2,
            num_actions=3,
            noisy=False,
        )
        options.update(kwargs)
        return RainbowQValueNetwork(**options)

    def test_dueling_uses_full_catalog_before_query_gather(self) -> None:
        network = self._network()
        with torch.no_grad():
            network.value_head.weight.zero_()
            network.value_head.bias.copy_(torch.tensor([10.0, 20.0]))
            network.advantage_head.weight.zero_()
            network.advantage_head.bias.copy_(
                torch.tensor([1.0, 5.0, 3.0, 9.0, 8.0, 1.0])
            )
        states = torch.zeros(2, 2)
        expected = torch.tensor([[7.0, 20.0], [9.0, 24.0], [14.0, 16.0]])
        all_quantiles = network(states)
        torch.testing.assert_close(all_quantiles, expected.expand(2, -1, -1))
        torch.testing.assert_close(
            all_quantiles.mean(dim=1), torch.tensor([[10.0, 20.0]]).expand(2, -1)
        )

        # Single queries retain the advantage, including action 2's identity.
        single = network.get_q_value_distribution(states, torch.tensor([[2], [0]]))
        torch.testing.assert_close(single, expected[torch.tensor([2, 0])])
        queries = torch.tensor([[[2], [0]], [[1], [1]]])
        subset = network.get_q_value_distribution(states, queries)
        torch.testing.assert_close(subset, expected[queries.squeeze(-1)])
        torch.testing.assert_close(
            network.get_q_values(states, queries), subset.mean(dim=-1)
        )

    def test_dueling_gradients_include_unqueried_advantages(self) -> None:
        network = self._network(num_quantiles=1)
        network.get_q_value_distribution(
            torch.zeros(1, 2), torch.tensor([[0]])
        ).sum().backward()
        torch.testing.assert_close(network.value_head.bias.grad, torch.ones(1))
        torch.testing.assert_close(
            network.advantage_head.bias.grad,
            torch.tensor([2 / 3, -1 / 3, -1 / 3]),
        )

    def test_plain_head_and_integral_float_actions(self) -> None:
        network = self._network(dueling=False)
        with torch.no_grad():
            network.quantile_head.weight.zero_()
            network.quantile_head.bias.copy_(torch.arange(6, dtype=torch.float32))
        states = torch.zeros(2, 2)
        values = network.get_q_value_distribution(states, torch.tensor([[2.0], [1.0]]))
        torch.testing.assert_close(values, torch.tensor([[4.0, 5.0], [2.0, 3.0]]))

    def test_invalid_queries_are_rejected(self) -> None:
        network = self._network()
        states = torch.zeros(1, 2)
        invalid = (
            torch.tensor([0]),
            torch.tensor([[0, 1]]),
            torch.tensor([[0.5]]),
            torch.tensor([[float("nan")]]),
            torch.tensor([[-1]]),
            torch.tensor([[3]]),
            torch.tensor([[0], [1]]),
        )
        for actions in invalid:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                network.get_q_value_distribution(states, actions)

    def test_network_noise_toggle_and_evaluation(self) -> None:
        network = self._network(hidden_dims=[8], noisy=True)
        states = torch.ones(2, 2)
        first = network(states)
        network.reset_noise()
        self.assertFalse(torch.equal(first, network(states)))
        network.enable_noise(False)
        expected = network(states)
        network.reset_noise()
        torch.testing.assert_close(network(states), expected)
        network.enable_noise(True)
        network.eval()
        torch.testing.assert_close(network(states), expected)
        network.reset_noise()
        torch.testing.assert_close(network(states), expected)

    def test_checkpoint_preserves_quantiles_and_sampled_noise(self) -> None:
        network = self._network(hidden_dims=[8], noisy=True).double()
        states = torch.ones(2, 2, dtype=torch.float64)
        restored = self._network(hidden_dims=[8], noisy=True).double()
        restored.load_state_dict(copy.deepcopy(network.state_dict()))
        torch.testing.assert_close(restored(states), network(states), rtol=0, atol=0)
        torch.testing.assert_close(
            restored.quantiles, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            restored.quantile_midpoints,
            torch.tensor([[[0.25, 0.75]]], dtype=torch.float64),
        )
        self.assertEqual(restored.state_dim, 2)
        self.assertEqual(restored.action_dim, 1)
        self.assertEqual(restored.num_actions, 3)
        self.assertEqual(restored.num_quantiles, 2)

    def test_empty_query_set_preserves_shapes(self) -> None:
        network = self._network()
        values = network.get_q_value_distribution(
            torch.zeros(2, 2), torch.empty(2, 0, 1, dtype=torch.long)
        )
        self.assertEqual(values.shape, (2, 0, 2))


if __name__ == "__main__":
    unittest.main()
