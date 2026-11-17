# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent mathematical oracles and PEARL integration regressions for C51."""

import copy
import math
import unittest
from unittest.mock import patch

import torch
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.common.noisy_linear import NoisyLinear
from pearl.neural_networks.sequential_decision_making.categorical_rainbow_q_value_network import (
    CategoricalRainbowQValueNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.sequential_decision_making.categorical_rainbow import (
    CategoricalRainbow,
    categorical_projection,
)
from pearl.replay_buffers.sequential_decision_making.prioritized_n_step_replay_buffer import (
    PrioritizedNStepReplayBuffer,
    PrioritizedNStepTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.risk_sensitive_safety_modules import RiskNeutralSafetyModule
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


def catalog(n=2):
    return DiscreteActionSpace([torch.tensor([i]) for i in range(n)])


def model(**kwargs):
    defaults = dict(
        state_dim=2,
        action_space=catalog(),
        hidden_dims=[],
        num_atoms=3,
        value_min=-1.0,
        value_max=1.0,
        noisy=False,
        dueling=False,
        learning_rate=0.01,
        target_update_freq=2,
    )
    defaults.update(kwargs)
    return CategoricalRainbow(**defaults)


def terminal_batch():
    return TransitionBatch(
        state=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        action=torch.tensor([[0], [1]]),
        reward=torch.tensor([0.25, -0.5]),
        terminated=torch.ones(2, dtype=torch.bool),
        weight=torch.tensor([0.25, 1.0]),
    )


class TestCategoricalProjection(unittest.TestCase):
    def test_exact_bins_terminal_fractional_and_clipped_atoms(self):
        support = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        probabilities = torch.tensor(
            [[0.2, 0.3, 0.5]] * 5, dtype=torch.float64, requires_grad=True
        )
        result = categorical_projection(
            probabilities,
            torch.tensor([0.0, 0.25, -4.0, 4.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5]),
            support,
        )
        expected = torch.tensor(
            [
                [0.2, 0.3, 0.5],
                [0.0, 0.75, 0.25],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.65, 0.25],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(result.sum(1), torch.ones(5, dtype=torch.float64))

    def test_projection_matches_scalar_hat_function_oracle(self):
        generator = torch.Generator().manual_seed(18)
        support = torch.linspace(-5, 7, 13, dtype=torch.float64)
        p = torch.rand(30, 13, generator=generator, dtype=torch.float64)
        p /= p.sum(1, keepdim=True)
        rewards = torch.randn(30, generator=generator, dtype=torch.float64) * 4
        discounts = torch.rand(30, generator=generator, dtype=torch.float64)
        discounts[:2] = torch.tensor([0.0, 1.0])
        expected = torch.zeros_like(p)
        # Independent triangular-basis definition, with no floor/ceil/scatter.
        for row in range(30):
            for atom in range(13):
                value = min(
                    7.0, max(-5.0, float(rewards[row] + discounts[row] * support[atom]))
                )
                for dest in range(13):
                    expected[row, dest] += p[row, atom] * max(
                        0.0, 1.0 - abs(float(support[dest]) - value)
                    )
        actual = categorical_projection(p, rewards, discounts, support)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), torch.ones(30, dtype=torch.float64))

    def test_invalid_distributions_and_support(self):
        for p, reward, discount, support in [
            (
                torch.tensor([[0.2, 0.2, 0.2]]),
                torch.tensor([0.0]),
                torch.tensor([1.0]),
                torch.tensor([-1.0, 0.0, 1.0]),
            ),
            (
                torch.tensor([[0.2, 0.3, 0.5]]),
                torch.tensor([float("nan")]),
                torch.tensor([1.0]),
                torch.tensor([-1.0, 0.0, 1.0]),
            ),
            (
                torch.tensor([[0.2, 0.3, 0.5]]),
                torch.tensor([0.0]),
                torch.tensor([-1.0]),
                torch.tensor([-1.0, 0.0, 1.0]),
            ),
            (
                torch.tensor([[0.2, 0.3, 0.5]]),
                torch.tensor([0.0]),
                torch.tensor([1.0]),
                torch.tensor([-1.0, 0.0, 2.0]),
            ),
        ]:
            with self.subTest(p=p, reward=reward, discount=discount, support=support):
                with self.assertRaises(ValueError):
                    categorical_projection(p, reward, discount, support)


class TestCategoricalNetwork(unittest.TestCase):
    def test_expected_values_and_subset_invariance(self):
        for dueling in [False, True]:
            net = CategoricalRainbowQValueNetwork(
                2,
                3,
                [],
                num_atoms=3,
                value_min=-2.0,
                value_max=4.0,
                dueling=dueling,
                noisy=False,
            )
            state = torch.randn(2, 2)
            all_actions = torch.tensor([[[0], [1], [2]]] * 2)
            subset = torch.tensor([[[2], [0]]] * 2)
            all_prob = net.get_probabilities(state, all_actions)
            torch.testing.assert_close(
                net.get_probabilities(state, subset), all_prob[:, [2, 0]]
            )
            expected = (all_prob * torch.tensor([-2.0, 1.0, 4.0])).sum(-1)
            torch.testing.assert_close(net.get_q_values(state, all_actions), expected)
            torch.testing.assert_close(all_prob.sum(-1), torch.ones(2, 3))
            self.assertEqual(
                net.get_logits(state, torch.tensor([[0], [1]])).shape, (2, 3)
            )
        with self.assertRaises(ValueError):
            net.get_logits(state, torch.tensor([[True], [False]]))
        with self.assertRaises(ValueError):
            net.get_logits(state, torch.tensor([[0.5], [1.0]]))

    def test_invalid_constructor(self):
        for kwargs in [
            dict(num_atoms=1),
            dict(num_atoms=2.5),
            dict(state_dim=True),
            dict(value_min=2, value_max=1),
            dict(value_max=float("inf")),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                args = dict(state_dim=2, num_actions=2, hidden_dims=[])
                args.update(kwargs)
                CategoricalRainbowQValueNetwork(**args)


class TestCategoricalLearner(unittest.TestCase):
    def test_weighted_cross_entropy_gradient_and_unweighted_kl_priority(self):
        learner = model(learning_rate=0.0, max_grad_norm=None)
        with torch.no_grad():
            learner._Q.categorical_head.weight.zero_()
            learner._Q.categorical_head.bias.zero_()
        batch = terminal_batch()
        target = torch.tensor([[0.0, 0.75, 0.25], [0.5, 0.5, 0.0]])
        report, priorities = learner._learn_batch_with_priorities(batch)
        self.assertAlmostEqual(report["loss"], 0.625 * math.log(3), places=6)
        expected_kl = torch.tensor(
            [
                math.log(3) + 0.75 * math.log(0.75) + 0.25 * math.log(0.25),
                math.log(3) + math.log(0.5),
            ]
        )
        torch.testing.assert_close(priorities, expected_kl)
        grad = (torch.ones(2, 3) / 3 - target) * batch.weight[:, None] / 2
        expected_bias = torch.cat([grad[0], grad[1]])
        torch.testing.assert_close(learner._Q.categorical_head.bias.grad, expected_bias)
        expected_weight = torch.zeros(6, 2)
        expected_weight[:3, 0] = grad[0]
        expected_weight[3:, 1] = grad[1]
        torch.testing.assert_close(
            learner._Q.categorical_head.weight.grad, expected_weight
        )
        self.assertTrue(all(p.grad is None for p in learner._Q_target.parameters()))

    def test_double_q_mask_order_terminal_nan_and_single_q(self):
        learner = model()
        online = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        target = torch.tensor([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
        with torch.no_grad():
            for net, p in [(learner._Q, online), (learner._Q_target, target)]:
                net.categorical_head.weight.zero_()
                net.categorical_head.bias.copy_(p.log().reshape(-1))
        batch = TransitionBatch(
            state=torch.zeros(2, 2),
            action=torch.tensor([[0], [1]]),
            reward=torch.tensor([0.0, 0.25]),
            terminated=torch.tensor([False, True]),
            next_state=torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]]),
            next_available_actions=torch.tensor(
                [[[1], [0], [-99]], [[-99], [-99], [-99]]]
            ),
            next_unavailable_actions_mask=torch.tensor(
                [[False, False, True], [True, True, True]]
            ),
        )
        actual = learner._next_probabilities(batch)
        torch.testing.assert_close(actual[0], target[1])
        learner._double_q = False
        torch.testing.assert_close(learner._next_probabilities(batch)[0], target[0])
        batch.next_unavailable_actions_mask[0, 1] = True
        torch.testing.assert_close(learner._next_probabilities(batch)[0], target[1])
        self.assertTrue(math.isfinite(learner.learn_batch(batch)["loss"]))
        batch.next_unavailable_actions_mask[0, 0] = True
        with self.assertRaises(ValueError):
            learner._next_probabilities(batch)

    def test_optimizer_step_counter_target_update_and_checkpoint(self):
        torch.manual_seed(4)
        learner = model()
        batch = terminal_batch()
        initial = copy.deepcopy(learner._Q_target.state_dict())
        learner.learn_batch(copy.deepcopy(batch))
        self.assertEqual(learner._training_steps, 1)
        for key, value in initial.items():
            torch.testing.assert_close(value, learner._Q_target.state_dict()[key])
        learner.learn_batch(copy.deepcopy(batch))
        self.assertEqual(learner._training_steps, 2)
        for p, q in zip(learner._Q.parameters(), learner._Q_target.parameters()):
            torch.testing.assert_close(p, q)
        restored = model()
        restored.load_state_dict(copy.deepcopy(learner.state_dict()))
        self.assertEqual(restored._training_steps, 2)
        self.assertEqual(
            learner.learn_batch(copy.deepcopy(batch)),
            restored.learn_batch(copy.deepcopy(batch)),
        )
        for p, q in zip(learner._Q.parameters(), restored._Q.parameters()):
            torch.testing.assert_close(p, q)
        with self.assertRaises(ValueError):
            model(value_max=2.0).load_state_dict(learner.state_dict())

    def test_terminal_history_and_trainable_parameter_ownership(self):
        history = LSTMHistorySummarizationModule(
            observation_dim=2, action_dim=1, history_length=3, hidden_dim=4
        )
        learner = model(state_dim=4, hidden_dims=[8])
        agent = PearlAgent(learner, history_summarization_module=history)
        batch = TransitionBatch(
            state=torch.rand(2, 3, 3),
            action=torch.tensor([[0], [1]]),
            reward=torch.tensor([0.25, -0.5]),
            terminated=torch.ones(2, dtype=torch.bool),
        ).to(agent.device)
        self.assertTrue(math.isfinite(agent.learn_batch(batch)["loss"]))
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in history.parameters()
            )
        )
        owned = [p for group in learner.optimizer.param_groups for p in group["params"]]
        for p in history.parameters():
            self.assertEqual(sum(q is p for q in owned), 1)

    def test_warmup_actions_and_noise_restoration_on_exception(self):
        learner = model(
            noisy=True,
            dueling=True,
            exploration_module=Warmup(EGreedyExploration(0.0), warmup_steps=2),
        )
        agent = PearlAgent(learner)
        agent.reset(torch.tensor([1.0, 0.0]), catalog())
        for _ in range(4):
            self.assertIn(int(agent.act().item()), [0, 1])
        layers = [m for m in learner._Q.modules() if isinstance(m, NoisyLinear)]
        layers[0].enable_noise(False)
        switches = [m.noise_enabled for m in layers]
        with patch.object(
            learner._Q, "get_q_values", side_effect=RuntimeError("injected")
        ):
            with self.assertRaises(RuntimeError):
                agent.act(exploit=True)
        self.assertEqual(switches, [m.noise_enabled for m in layers])
        first = agent.act(exploit=True)
        for _ in range(3):
            self.assertTrue(torch.equal(first, agent.act(exploit=True)))
        learner.safety_module = RiskNeutralSafetyModule()
        with self.assertRaises(TypeError):
            agent.act()

    def test_prioritized_nstep_native_agent_integration(self):
        replay = PrioritizedNStepReplayBuffer(32, n_steps=3, discount_factor=0.99)
        learner = model(batch_size=2)
        agent = PearlAgent(learner, replay_buffer=replay)
        for i in range(4):
            replay.push(
                state=torch.tensor([float(i), 0.0]),
                action=torch.tensor([i % 2]),
                reward=0.25,
                next_state=torch.tensor([float(i + 1), 0.0]),
                curr_available_actions=catalog(),
                next_available_actions=catalog(),
                terminated=i == 3,
                truncated=False,
                max_number_actions=2,
            )
        self.assertEqual(len(replay), 4)
        with patch.object(
            replay, "update_priorities", wraps=replay.update_priorities
        ) as update:
            self.assertIn("loss", agent.learn())
            self.assertEqual(update.call_count, 1)
            self.assertTrue((update.call_args.args[1] >= 0).all())
        batch = replay.sample(4)
        self.assertIsInstance(batch, PrioritizedNStepTransitionBatch)
        replay.discount_factor = 0.9
        with self.assertRaises(ValueError):
            learner.learn(replay)

    def test_terminal_learning_improves_fixed_distribution_fit(self):
        torch.manual_seed(20)
        learner = model(learning_rate=0.03)
        losses = [learner.learn_batch(terminal_batch())["loss"] for _ in range(100)]
        self.assertLess(losses[-1], losses[0] - 0.1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA device")
    def test_cuda_projection_and_one_optimizer_update_match_cpu(self):
        torch.manual_seed(13)
        cpu = model()
        gpu = model().cuda()
        gpu.load_state_dict(copy.deepcopy(cpu.state_dict()))
        batch = terminal_batch()
        a = cpu.learn_batch(copy.deepcopy(batch))
        b = gpu.learn_batch(copy.deepcopy(batch).to(torch.device("cuda")))
        self.assertAlmostEqual(a["loss"], b["loss"], places=5)
        for p, q in zip(cpu._Q.parameters(), gpu._Q.parameters()):
            torch.testing.assert_close(p, q.cpu(), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
