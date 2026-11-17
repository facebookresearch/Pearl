# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Scalar lambda-return oracles and native/vector integration regressions."""

import copy
import io
import unittest

import torch
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.parallel_q_value_network import (
    ParallelQValueNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.parallel_q_learning import (
    ParallelQLearning,
    q_lambda_returns,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.sequential_decision_making.pqn_rollout_buffer import (
    PQNRollout,
    PQNRolloutBuffer,
    PQNTransitionBatch,
)
from pearl.safety_modules.risk_sensitive_safety_modules import RiskNeutralSafetyModule
from pearl.utils.functional_utils.train_and_eval.pqn_vector_collector import (
    PQNVectorCollector,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


def catalog(ids=(0, 1)):
    return DiscreteActionSpace([torch.tensor([value]) for value in ids])


def learner(**kwargs):
    options = dict(
        state_dim=2,
        action_space=catalog(),
        hidden_dims=[8],
        learning_rate=0.01,
        update_epochs=2,
        batch_size=4,
    )
    options.update(kwargs)
    return ParallelQLearning(**options)


class AutoresetExample(Environment):
    """The advertised observation is intentionally wrong for the timeout bootstrap."""

    def __init__(self, truncated=True):
        self.resets = 0
        self.truncated = truncated

    @property
    def action_space(self):
        return catalog()

    @property
    def observation_space(self):
        return None

    def reset(self, seed=None):
        self.resets += 1
        return torch.tensor([100.0 * self.resets, 0.0]), self.action_space

    def step(self, action):
        return ActionResult(
            torch.tensor([999.0, 0.0]),
            reward=1.0,
            terminated=not self.truncated,
            truncated=self.truncated,
            info={"final_observation": torch.tensor([3.0, 0.0])},
        )


class TestPQNReturns(unittest.TestCase):
    def test_integer_rewards_preserve_fractional_bootstrap_and_lambda_returns(self):
        rewards = torch.tensor([[1], [2]])
        values = torch.tensor([[1.5], [3.5]])
        flags = torch.zeros(2, 1, dtype=torch.bool)
        actual = q_lambda_returns(rewards, values, flags, flags, 0.9, 0.0)
        torch.testing.assert_close(actual, torch.tensor([[2.35], [5.15]]))
        self.assertTrue(actual.is_floating_point())
        all_integer = q_lambda_returns(
            rewards, torch.tensor([[2], [3]]), flags, flags, 0.9, 0.0
        )
        torch.testing.assert_close(all_integer, torch.tensor([[2.8], [4.7]]))

    def test_integer_rollout_rewards_use_network_prediction_dtype(self):
        model = learner(hidden_dims=[], discount_factor=0.9, trace_decay=0.0)
        with torch.no_grad():
            model._Q.network[-1].weight.zero_()
            model._Q.network[-1].bias.copy_(torch.tensor([1.5, 3.5]))
        rollout = PQNRollout(
            states=torch.zeros(2, 1, 2),
            actions=torch.zeros(2, 1, 1, dtype=torch.long),
            rewards=torch.tensor([[1], [2]]),
            terminated=torch.zeros(2, 1, dtype=torch.bool),
            truncated=torch.zeros(2, 1, dtype=torch.bool),
            next_states=torch.zeros(2, 1, 2),
        )
        torch.testing.assert_close(
            model.rollout_targets(rollout), torch.tensor([[4.15], [5.15]])
        )

    def test_lambda_returns_match_forward_nstep_mixture_oracle(self):
        torch.manual_seed(33)
        rewards = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        values = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        terminal = torch.zeros(7, 3, dtype=torch.bool)
        truncation = terminal.clone()
        terminal[2, 0] = True
        truncation[3, 1] = True
        terminal[6, 2] = True
        for decay in (0.0, 0.2, 0.65, 1.0):
            expected = torch.zeros_like(rewards)
            # Enumerate each n-step return and mix forward; no backward recursion.
            for env in range(3):
                for start in range(7):
                    end = start
                    while end < 6 and not (terminal[end, env] or truncation[end, env]):
                        end += 1
                    horizon = end - start + 1
                    for length in range(1, horizon + 1):
                        final = start + length - 1
                        value = sum(
                            (0.9**offset) * float(rewards[start + offset, env].detach())
                            for offset in range(length)
                        )
                        if not terminal[final, env]:
                            value += (0.9**length) * float(values[final, env].detach())
                        weight = (decay ** (length - 1)) * (
                            (1 - decay) if length < horizon else 1.0
                        )
                        expected[start, env] += weight * value
            result = q_lambda_returns(rewards, values, terminal, truncation, 0.9, decay)
            torch.testing.assert_close(result, expected)
            self.assertFalse(result.requires_grad)

    def test_terminal_nan_is_not_multiplied_by_zero_and_timeout_bootstraps(self):
        result = q_lambda_returns(
            torch.tensor([[1.0, 1.0], [100.0, 100.0]]),
            torch.tensor([[float("nan"), 3.0], [4.0, 4.0]]),
            torch.tensor([[True, False], [False, False]]),
            torch.tensor([[False, True], [False, False]]),
            0.9,
            0.65,
        )
        torch.testing.assert_close(result[0], torch.tensor([1.0, 3.7]))

    def test_rollout_rejects_autoreset_placeholder_at_timeout(self):
        with self.assertRaises(ValueError):
            PQNRollout(
                states=torch.zeros(1, 1, 2),
                actions=torch.zeros(1, 1, 1, dtype=torch.long),
                rewards=torch.ones(1, 1),
                terminated=torch.zeros(1, 1, dtype=torch.bool),
                truncated=torch.ones(1, 1, dtype=torch.bool),
                next_states=torch.full((1, 1, 2), float("nan")),
            )


class TestPQNNetworkAndLearner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(54)

    def test_native_q_interface_and_masks(self):
        network = ParallelQValueNetwork(2, 3, [8])
        states = torch.rand(2, 2)
        full = network(states)
        ids = torch.tensor([[[2], [0]], [[2], [0]]])
        torch.testing.assert_close(network.get_q_values(states, ids), full[:, [2, 0]])
        for action in (
            torch.tensor([[True], [False]]),
            torch.tensor([[0.5], [1.0]]),
            torch.tensor([[0], [3]]),
        ):
            with self.assertRaises(ValueError):
                network.get_q_values(states, action)
        model = learner(action_space=catalog((0, 1, 2)), hidden_dims=[])
        with torch.no_grad():
            model._Q.network[-1].weight.zero_()
            model._Q.network[-1].bias.copy_(torch.tensor([1.0, 20.0, -2.0]))
        mask = torch.tensor([[False, True, False], [True, True, False]])
        self.assertEqual(
            model.act_batch(states, mask, exploit=True).tolist(), [[0], [2]]
        )
        draws = model.act_batch(states.repeat(50, 1), mask.repeat(50, 1), epsilon=1.0)
        self.assertTrue(torch.isin(draws[::2], torch.tensor([0, 2])).all())
        self.assertTrue((draws[1::2] == 2).all())
        before = model._environment_steps
        self.assertEqual(
            int(model.act(torch.ones(2), catalog((2,)), exploit=True).item()), 2
        )
        self.assertEqual(model._environment_steps, before)
        self.assertFalse(hasattr(model, "_Q_target"))

    def test_analytical_mse_gradient_and_detached_target(self):
        model = learner(hidden_dims=[], learning_rate=0.0, max_grad_norm=None)
        with torch.no_grad():
            model._Q.network[-1].weight.zero_()
            model._Q.network[-1].bias.zero_()
        targets = torch.tensor([1.0, -2.0], requires_grad=True)
        batch = PQNTransitionBatch(
            state=torch.eye(2),
            action=torch.tensor([[0], [1]]),
            reward=torch.zeros(2),
            returns=targets,
        )
        report = model.learn_batch(batch)
        self.assertAlmostEqual(report["loss"], 2.5)
        torch.testing.assert_close(
            model._Q.network[-1].bias.grad, torch.tensor([-1.0, 2.0])
        )
        torch.testing.assert_close(
            model._Q.network[-1].weight.grad, torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        )
        self.assertIsNone(targets.grad)

    def test_targets_skip_terminal_network_and_apply_next_action_mask(self):
        model = learner(hidden_dims=[])
        with torch.no_grad():
            model._Q.network[-1].weight.zero_()
            model._Q.network[-1].bias.copy_(torch.tensor([1.0, 100.0]))
        rollout = PQNRollout(
            states=torch.zeros(1, 2, 2),
            actions=torch.zeros(1, 2, 1, dtype=torch.long),
            rewards=torch.tensor([[2.0, 3.0]]),
            terminated=torch.tensor([[False, True]]),
            truncated=torch.zeros(1, 2, dtype=torch.bool),
            next_states=torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]]]),
            next_unavailable_actions_mask=torch.tensor([[[False, True], [True, True]]]),
        )
        targets = model.rollout_targets(rollout)
        torch.testing.assert_close(targets, torch.tensor([[2.99, 3.0]]))
        self.assertFalse(targets.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_native_agent_consumes_fresh_rollout_without_persistent_replay(self):
        model = learner()
        replay = PQNRolloutBuffer(max_steps=4)
        agent = PearlAgent(model, replay_buffer=replay)
        agent.reset(torch.tensor([0.0, 1.0]), catalog())
        before = [p.detach().clone() for p in model._Q.parameters()]
        for index in range(4):
            agent.act()
            agent.observe(
                ActionResult(torch.tensor([index * 0.1, 1.0]), 1.0, index == 3, False)
            )
        report = agent.learn()
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual(len(replay), 0)
        self.assertTrue(
            any(not torch.equal(p, q) for p, q in zip(before, model._Q.parameters()))
        )
        with self.assertRaises(TypeError):
            model.learn(BasicReplayBuffer(4))

    def test_checkpoint_restores_optimizer_and_counters(self):
        model = learner()
        batch = PQNTransitionBatch(
            state=torch.eye(2),
            action=torch.tensor([[0], [1]]),
            reward=torch.zeros(2),
            returns=torch.tensor([1.0, -2.0]),
        )
        model.learn_batch(copy.deepcopy(batch))
        model.act_batch(torch.eye(2), epsilon=1.0)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = learner()
        restored.load_state_dict(
            torch.load(stream, weights_only=True, map_location="cpu")
        )
        self.assertEqual(restored._training_steps, 1)
        self.assertEqual(restored._environment_steps, 2)
        self.assertEqual(
            model.learn_batch(copy.deepcopy(batch)),
            restored.learn_batch(copy.deepcopy(batch)),
        )
        for p, q in zip(model._Q.parameters(), restored._Q.parameters()):
            torch.testing.assert_close(p, q)

    def test_explicit_unsupported_components(self):
        model = learner()
        with self.assertRaises(TypeError):
            model.set_history_summarization_module(
                LSTMHistorySummarizationModule(
                    observation_dim=2, action_dim=1, history_length=2, hidden_dim=2
                )
            )
        model.safety_module = RiskNeutralSafetyModule()
        with self.assertRaises(TypeError):
            model.act_batch(torch.ones(1, 2))

    def test_invalid_hyperparameters(self):
        for kwargs in (
            dict(update_epochs=True),
            dict(batch_size=2.5),
            dict(epsilon=float("nan")),
            dict(trace_decay=1.1),
            dict(discount_factor=-0.1),
            dict(weight_decay=-1.0),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                learner(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA")
    def test_populated_optimizer_checkpoint_moves_cpu_cuda_cpu(self):
        cpu = learner(hidden_dims=[])
        batch = PQNTransitionBatch(
            state=torch.eye(2),
            action=torch.tensor([[0], [1]]),
            reward=torch.zeros(2),
            returns=torch.tensor([1.0, -2.0]),
        )
        cpu.learn_batch(copy.deepcopy(batch))
        gpu = learner(hidden_dims=[]).cuda()
        gpu.load_state_dict(copy.deepcopy(cpu.state_dict()))
        a = cpu.learn_batch(copy.deepcopy(batch))["loss"]
        b = gpu.learn_batch(copy.deepcopy(batch).to(torch.device("cuda")))["loss"]
        self.assertAlmostEqual(a, b, places=5)
        for p, state in gpu.optimizer.state.items():
            self.assertEqual(state["exp_avg"].device, p.device)
        restored = learner(hidden_dims=[])
        restored.load_state_dict(copy.deepcopy(gpu.state_dict()))
        for p, state in restored.optimizer.state.items():
            self.assertEqual(state["exp_avg"].device, p.device)
        for p, q in zip(cpu._Q.parameters(), restored._Q.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-5)


class TestPQNVectorCollector(unittest.TestCase):
    def test_final_observation_bootstrap_reset_separation_and_actual_step_count(self):
        model = learner(hidden_dims=[])
        with torch.no_grad():
            model._Q.network[-1].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
            model._Q.network[-1].bias.zero_()
        environments = [AutoresetExample(), AutoresetExample(truncated=False)]
        collector = PQNVectorCollector(environments, model, seed=42)
        times = []
        rollout = collector.collect(3, epsilon=lambda step: (times.append(step) or 0.0))
        self.assertEqual(collector.steps, 6)
        self.assertEqual(times, [0, 2, 4])
        self.assertEqual(rollout.states[:, 0, 0].tolist(), [100.0, 200.0, 300.0])
        self.assertEqual(rollout.next_states[:, 0, 0].tolist(), [3.0, 3.0, 3.0])
        targets = model.rollout_targets(rollout)
        torch.testing.assert_close(targets[:, 0], torch.full((3,), 3.97))
        torch.testing.assert_close(targets[:, 1], torch.ones(3))
        self.assertEqual(len(collector.episodes), 6)
        self.assertEqual(collector.snapshot()["incomplete_episode_lengths"], [0, 0])
        report = model.learn_rollout(rollout)
        self.assertEqual(report["transitions"], 6)
        self.assertEqual(report["optimizer_updates"], 4)
        collector.close()


if __name__ == "__main__":
    unittest.main()
