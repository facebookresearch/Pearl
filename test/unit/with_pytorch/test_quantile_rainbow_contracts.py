# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent numeric oracles and native PEARL lifecycle regressions."""

import copy
import io
import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action_result import ActionResult
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.common.noisy_linear import NoisyLinear
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.sequential_decision_making.quantile_rainbow import (
    QuantileRainbow,
)
from pearl.replay_buffers.sequential_decision_making.prioritized_n_step_replay_buffer import (
    PrioritizedNStepReplayBuffer,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

DEVICE = torch.device(os.environ.get("PEARL_TEST_DEVICE", "cpu"))


def space(ids=(0, 1, 2)):
    return DiscreteActionSpace([torch.tensor([a]) for a in ids])


def learner(**options):
    settings = dict(
        state_dim=2,
        action_space=space(),
        hidden_dims=[8],
        num_quantiles=4,
        noisy=True,
        learning_rate=0.01,
        target_update_freq=2,
    )
    settings.update(options)
    return QuantileRainbow(**settings).to(DEVICE)


def push(replay, value=0, terminated=False, truncated=False, reward=1.0):
    replay.push(
        state=torch.tensor([float(value), 1.0]),
        action=torch.tensor([0]),
        reward=reward,
        next_state=torch.tensor([float(value + 1), 1.0]),
        terminated=terminated,
        truncated=truncated,
        curr_available_actions=space(),
        next_available_actions=space(),
        max_number_actions=3,
    )


class TestQuantileRainbowContracts(unittest.TestCase):
    def test_native_warmup_wrapper_returns_scalar_actions(self):
        exploration = Warmup(EGreedyExploration(0.0), warmup_steps=2)
        model = learner(exploration_module=exploration)
        for _ in range(4):
            action = model.act(torch.ones(2, device=DEVICE), space())
            self.assertEqual(action.numel(), 1)
            self.assertIn(int(action.item()), (0, 1, 2))
        counter = exploration.time_step
        model.act(torch.ones(2, device=DEVICE), space(), exploit=True)
        self.assertEqual(exploration.time_step, counter)

    def setUp(self):
        torch.manual_seed(123)
        if DEVICE.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.25)

    def test_native_explicit_identity_action_representation_hook(self):
        module = IdentityActionRepresentationModule(
            max_number_actions=3, representation_dim=1
        )
        model = learner(action_representation_module=module)
        self.assertIs(model.action_representation_module, module)
        self.assertEqual(
            int(model.act(torch.zeros(2, device=DEVICE), space((2,)), exploit=True)), 2
        )

    def test_quantile_loss_and_gradient_match_scalar_oracle(self):
        predicted = [[-2.0, 0.25, 0.5, 3.0], [1.0, -0.5, 2.0, 4.0]]
        targets = [[-1.0, 0.5, 2.0], [-2.0, 0.0, 5.0]]
        expected, gradients = [], []
        for estimates, outcomes in zip(predicted, targets):
            loss, gradient_row = 0.0, []
            for i, estimate in enumerate(estimates):
                tau, grad = (i + 0.5) / len(estimates), 0.0
                for outcome in outcomes:
                    residual = outcome - estimate
                    asymmetry = abs(tau - float(residual < 0))
                    magnitude = abs(residual)
                    loss += (
                        asymmetry
                        * (0.5 * residual**2 if magnitude <= 1 else magnitude - 0.5)
                        / len(outcomes)
                    )
                    grad -= asymmetry * max(-1.0, min(1.0, residual)) / len(outcomes)
                gradient_row.append(grad)
            expected.append(loss)
            gradients.append(gradient_row)
        pred = torch.tensor(
            predicted, dtype=torch.float64, device=DEVICE, requires_grad=True
        )
        target = torch.tensor(targets, dtype=torch.float64, device=DEVICE)
        tau = (torch.arange(4, dtype=torch.float64, device=DEVICE) + 0.5) / 4
        loss = QuantileRainbow._per_transition_quantile_loss(pred, target, tau)
        torch.testing.assert_close(
            loss,
            torch.tensor(expected, dtype=torch.float64, device=DEVICE),
            rtol=1e-12,
            atol=1e-12,
        )
        loss.sum().backward()
        torch.testing.assert_close(
            pred.grad,
            torch.tensor(gradients, dtype=torch.float64, device=DEVICE),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_noisy_layer_dtype_gradients_resampling_and_eval_on_device(self):
        layer = NoisyLinear(5, 4).to(device=DEVICE, dtype=torch.float64)
        state = torch.arange(15, dtype=torch.float64, device=DEVICE).reshape(3, 5) / 10
        before = layer.weight_epsilon.clone()
        layer.reset_noise()
        self.assertFalse(torch.equal(before, layer.weight_epsilon))
        layer(state).square().mean().backward()
        for parameter in (
            layer.weight_mu,
            layer.weight_sigma,
            layer.bias_mu,
            layer.bias_sigma,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        layer.eval()
        expected = layer(state)
        layer.reset_noise()
        torch.testing.assert_close(layer(state), expected, rtol=0, atol=0)

    def test_online_target_noise_remains_independent_after_hard_update(self):
        model = learner(target_update_freq=1)
        batch = TransitionBatch(
            state=torch.ones(2, 2, device=DEVICE),
            action=torch.tensor([[0], [1]], device=DEVICE),
            reward=torch.tensor([1.0, 2.0], device=DEVICE),
            terminated=torch.ones(2, dtype=torch.bool, device=DEVICE),
        )
        model.learn_batch(batch)
        online = [m for m in model._Q.modules() if isinstance(m, NoisyLinear)]
        target = [m for m in model._Q_target.modules() if isinstance(m, NoisyLinear)]
        for a, b in zip(online, target):
            torch.testing.assert_close(a.weight_mu, b.weight_mu, rtol=0, atol=0)
            self.assertFalse(torch.equal(a.weight_epsilon, b.weight_epsilon))

    def test_stale_sample_rejected_after_ring_overwrite(self):
        replay = PrioritizedNStepReplayBuffer(2, n_steps=1)
        push(replay, 0)
        push(replay, 1)
        with patch("torch.multinomial", return_value=torch.tensor([0])):
            old = replay.sample(1)
        push(replay, 2)
        before = replay._priorities.clone()
        with self.assertRaisesRegex(ValueError, "stale"):
            replay.update_priorities(old.indices, torch.tensor([99.0]))
        torch.testing.assert_close(before, replay._priorities, rtol=0, atol=0)

    def test_stale_sample_rejected_after_clear_and_refill(self):
        replay = PrioritizedNStepReplayBuffer(2, n_steps=1)
        push(replay)
        old = replay.sample(1)
        replay.clear()
        push(replay, 10)
        replay.sample(1)  # A new sample must not make the old identity valid.
        with self.assertRaisesRegex(ValueError, "stale"):
            replay.update_priorities(old.indices, torch.tensor([99.0]))

    def test_new_sample_updates_correct_slot_after_wrap(self):
        replay = PrioritizedNStepReplayBuffer(2, n_steps=1)
        for i in range(5):
            push(replay, i)
        with patch("torch.multinomial", return_value=torch.tensor([0, 1, 0])):
            batch = replay.sample(3)
        replay.update_priorities(batch.indices, torch.tensor([3.0, 2.0, 7.0]))
        torch.testing.assert_close(
            replay._priorities, torch.tensor([7.000001, 2.000001], dtype=torch.float64)
        )

    def test_exploit_restores_each_noise_switch_after_exception(self):
        model = learner()
        layers = [m for m in model._Q.modules() if isinstance(m, NoisyLinear)]
        layers[0].enable_noise(False)
        switches = [m.noise_enabled for m in layers]
        modes = [m.training for m in model.modules()]
        eps = [m.weight_epsilon.clone() for m in layers]
        with patch.object(
            model.safety_module,
            "get_q_values_under_risk_metric",
            side_effect=RuntimeError("oracle exception"),
        ):
            with self.assertRaisesRegex(RuntimeError, "oracle exception"):
                model.act(torch.zeros(2, device=DEVICE), space(), exploit=True)
        self.assertEqual(switches, [m.noise_enabled for m in layers])
        self.assertEqual(modes, [m.training for m in model.modules()])
        for before, layer in zip(eps, layers):
            torch.testing.assert_close(before, layer.weight_epsilon, rtol=0, atol=0)

    def test_fractional_update_settings_rejected_at_construction(self):
        for options in [
            dict(target_update_freq=1.5),
            dict(training_rounds=1.5),
            dict(batch_size=1.5),
            dict(target_update_freq=True),
        ]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "integer"):
                    learner(**options)

    def test_bool_and_complex_action_ids_rejected_clearly(self):
        model = learner()
        for ids in [[True, False], [0 + 0j, 1 + 0j]]:
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "action|IDs|integer"):
                    model.act(torch.zeros(2, device=DEVICE), space(ids))

    def test_unsupported_trainable_action_representation_rejected(self):
        class TrainableIdentity(IdentityActionRepresentationModule):
            def __init__(self):
                super().__init__(max_number_actions=3, representation_dim=1)
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                return x * self.scale

        model = learner()
        model.action_representation_module = TrainableIdentity().to(DEVICE)
        with self.assertRaisesRegex(ValueError, "representation"):
            model.act(torch.zeros(2, device=DEVICE), space(), exploit=True)

    def test_n_step_returns_exhaustive_short_episodes_and_wrap(self):
        for n in (1, 2, 4):
            for gamma in (0.0, 0.5, 1.0):
                for length in (1, 2, 5):
                    for terminal in (False, True):
                        with self.subTest(
                            n=n, gamma=gamma, length=length, terminal=terminal
                        ):
                            replay = PrioritizedNStepReplayBuffer(
                                3, n_steps=n, discount_factor=gamma
                            )
                            for t in range(length):
                                push(
                                    replay,
                                    t,
                                    terminated=terminal and t == length - 1,
                                    truncated=not terminal and t == length - 1,
                                    reward=float(t + 1),
                                )
                            self.assertEqual(len(replay), min(length, 3))
                            self.assertEqual(len(replay._pending), 0)
                            for record in replay._storage:
                                start = int(record.transition.state[0, 0])
                                horizon = min(n, length - start)
                                reward = sum(
                                    (gamma**k) * (start + k + 1) for k in range(horizon)
                                )
                                discount = (
                                    0.0
                                    if terminal and start + horizon == length
                                    else gamma**horizon
                                )
                                self.assertAlmostEqual(
                                    float(record.transition.reward), reward, places=6
                                )
                                self.assertAlmostEqual(
                                    record.bootstrap_discount, discount, places=12
                                )
                                self.assertEqual(
                                    int(record.transition.next_state[0, 0]),
                                    start + horizon,
                                )

    def test_priorities_sampling_and_population_normalized_weights(self):
        replay = PrioritizedNStepReplayBuffer(
            3, n_steps=1, alpha=1.0, beta=1.0, priority_epsilon=1.0
        )
        for i in range(3):
            push(replay, i)
        replay.update_priorities(torch.tensor([0, 1, 2]), torch.tensor([0.0, 1.0, 6.0]))
        batch = replay.sample(20_000)
        counts = torch.bincount(batch.indices, minlength=3).double() / 20_000
        torch.testing.assert_close(
            counts,
            torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64),
            rtol=0,
            atol=0.015,
        )
        torch.testing.assert_close(
            batch.weight, torch.tensor([1.0, 0.5, 1 / 7])[batch.indices]
        )
        replay.beta = 0.0
        torch.testing.assert_close(replay.sample(5).weight, torch.ones(5))

    def test_replay_owns_push_and_sample_data(self):
        replay = PrioritizedNStepReplayBuffer(4, n_steps=1)
        state = np.array([2.0, 3.0], dtype=np.float32)
        action = torch.tensor([0])
        replay.push(
            state=state,
            action=action,
            reward=1.0,
            next_state=state,
            terminated=False,
            truncated=True,
            curr_available_actions=space(),
            next_available_actions=space(),
            max_number_actions=3,
        )
        state[:] = 99
        action[:] = 2
        batch = replay.sample(1)
        torch.testing.assert_close(batch.state, torch.tensor([[2.0, 3.0]]))
        self.assertEqual(int(batch.action[0, 0]), 0)
        batch.state[:] = 88
        batch.reward[:] = 88
        batch.next_available_actions[:] = 88
        again = replay.sample(1)
        torch.testing.assert_close(again.state, torch.tensor([[2.0, 3.0]]))
        self.assertEqual(float(again.reward[0]), 1.0)
        self.assertLess(float(again.next_available_actions.max()), 3.0)

    def test_native_trainable_history_is_owned_and_updated(self):
        history = LSTMHistorySummarizationModule(
            observation_dim=2,
            action_dim=1,
            history_length=2,
            hidden_dim=2,
            num_layers=1,
        )
        model = learner(noisy=False)
        agent = PearlAgent(
            policy_learner=model,
            replay_buffer=PrioritizedNStepReplayBuffer(8, n_steps=1),
            history_summarization_module=history,
            device_id=0 if DEVICE.type == "cuda" else -1,
        )
        before = [p.detach().clone() for p in history.parameters()]
        agent.reset(torch.ones(2), space())
        for t in range(4):
            agent.act()
            agent.observe(
                ActionResult(
                    observation=torch.tensor([1.0, float(t + 1)]),
                    reward=2.0,
                    terminated=t == 3,
                    truncated=False,
                )
            )
            agent.learn()
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, history.parameters()))
        )
        owned = [p for group in model._optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(owned), len({id(p) for p in owned}))
        self.assertTrue(
            all(id(p) in {id(x) for x in owned} for p in history.parameters())
        )

    def test_terminal_history_preprocessing_skips_missing_next_state(self):
        history = LSTMHistorySummarizationModule(
            observation_dim=2,
            action_dim=1,
            history_length=2,
            hidden_dim=2,
            num_layers=1,
        )
        agent = PearlAgent(
            policy_learner=learner(noisy=False),
            history_summarization_module=history,
            replay_buffer=PrioritizedNStepReplayBuffer(4),
            device_id=0 if DEVICE.type == "cuda" else -1,
        )
        batch = TransitionBatch(
            state=torch.ones(1, 2, 3, device=DEVICE),
            action=torch.tensor([[0]], device=DEVICE),
            reward=torch.tensor([1.0], device=DEVICE),
            terminated=torch.tensor([True], device=DEVICE),
            next_state=None,
        )
        calls = []
        handle = history.register_forward_pre_hook(
            lambda module, inputs: calls.append(inputs[0])
        )
        try:
            report = agent.learn_batch(batch)
        finally:
            handle.remove()
        self.assertTrue(math.isfinite(report["loss"]))
        self.assertEqual(len(calls), 1)

    def test_known_terminal_mdp_converges_to_reward_and_optimal_action(self):
        model = learner(
            hidden_dims=[], num_quantiles=5, noisy=False, learning_rate=0.03
        )
        batch = TransitionBatch(
            state=torch.tensor([[1.0, 0.0]] * 3, device=DEVICE),
            action=torch.tensor([[0], [1], [2]], device=DEVICE),
            reward=torch.tensor([-1.0, 0.0, 2.0], device=DEVICE),
            terminated=torch.ones(3, dtype=torch.bool, device=DEVICE),
        )
        losses = [model.learn_batch(batch)["loss"] for _ in range(250)]
        values = model._Q.get_q_values(batch.state, batch.action)
        torch.testing.assert_close(values, batch.reward, rtol=0, atol=0.12)
        self.assertLess(losses[-1], losses[0] * 0.02)
        self.assertEqual(
            int(model.act(batch.state[0], space(), exploit=True).item()), 2
        )
        self.assertEqual(model._training_steps, 250)

    @unittest.skipUnless(
        DEVICE.type == "cuda", "CPU/CUDA checkpoint movement requires the real CUDA row"
    )
    def test_checkpoint_moves_optimizer_between_cpu_and_cuda(self):
        model = learner(noisy=False)
        batch = TransitionBatch(
            state=torch.ones(3, 2, device=DEVICE),
            action=torch.tensor([[0], [1], [2]], device=DEVICE),
            reward=torch.tensor([-1.0, 0.0, 2.0], device=DEVICE),
            terminated=torch.ones(3, dtype=torch.bool, device=DEVICE),
        )
        model.learn_batch(batch)
        cpu = learner(noisy=False).cpu()
        cpu.load_state_dict(copy.deepcopy(model.state_dict()))
        gpu = learner(noisy=False)
        gpu.load_state_dict(copy.deepcopy(cpu.state_dict()))
        expected = model._Q.get_q_values(batch.state, batch.action)
        actual_cpu = cpu._Q.get_q_values(batch.state.cpu(), batch.action.cpu())
        torch.testing.assert_close(expected.cpu(), actual_cpu, rtol=1e-5, atol=1e-6)
        model.learn_batch(copy.deepcopy(batch))
        cpu.learn_batch(copy.deepcopy(batch).to(torch.device("cpu")))
        gpu.learn_batch(copy.deepcopy(batch))
        for a, b, c in zip(
            model._Q.parameters(), cpu._Q.parameters(), gpu._Q.parameters()
        ):
            torch.testing.assert_close(a.cpu(), b, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(a, c, rtol=1e-6, atol=1e-7)

    def test_native_agent_subset_reordering_gradients_and_checkpoint(self):
        replay = PrioritizedNStepReplayBuffer(7, n_steps=3)
        model = learner(training_rounds=2)
        agent = PearlAgent(
            policy_learner=model,
            replay_buffer=replay,
            device_id=0 if DEVICE.type == "cuda" else -1,
        )
        if DEVICE.type == "cuda":
            self.assertTrue(torch.cuda.is_available())
            self.assertEqual(next(model._Q.parameters()).device.type, "cuda")
        owned = [p for g in model._optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(owned), len({id(p) for p in owned}))
        self.assertEqual(
            {id(p) for p in owned},
            {id(p) for p in model.parameters() if p.requires_grad},
        )
        before = [p.detach().clone() for p in model._Q.parameters()]
        target_before = [p.detach().clone() for p in model._Q_target.parameters()]
        agent.reset(torch.zeros(2), space((2, 0)))
        for t in range(12):
            current = agent._action_space
            action = agent.act()
            self.assertIn(int(action.item()), [int(x.item()) for x in current.actions])
            final = t % 4 == 3
            nxt = space((1,)) if t % 2 == 0 else space((2, 0))
            agent.observe(
                ActionResult(
                    observation=torch.tensor([float(t + 1), 1.0]),
                    reward=float(t % 3),
                    terminated=False,
                    truncated=final,
                    available_action_space=nxt,
                )
            )
            agent.learn()
            if final:
                self.assertEqual(len(replay._pending), 0)
                agent.reset(torch.zeros(2), space((2, 0)))
        self.assertGreater(model._training_steps, 0)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model._Q.parameters()))
        )
        self.assertTrue(
            any(
                not torch.equal(a, b)
                for a, b in zip(target_before, model._Q_target.parameters())
            )
        )
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model._Q.parameters()
            )
        )
        self.assertTrue(all(p.grad is None for p in model._Q_target.parameters()))
        sampled = replay.sample(4)
        self.assertEqual(sampled.indices.device, agent.device)
        self.assertEqual(sampled.weight.device, agent.device)
        replay.update_priorities(sampled.indices, torch.ones(4, device=agent.device))
        checkpoint = io.BytesIO()
        torch.save(agent.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = PearlAgent(
            policy_learner=learner(training_rounds=2),
            replay_buffer=PrioritizedNStepReplayBuffer(7, n_steps=3),
            device_id=0 if DEVICE.type == "cuda" else -1,
        )
        restored.load_state_dict(
            torch.load(checkpoint, map_location=agent.device, weights_only=True)
        )
        observation = torch.tensor([0.5, 1.0])
        agent.reset(observation, space())
        restored.reset(observation, space())
        for _ in range(3):
            torch.testing.assert_close(
                agent.act(exploit=True), restored.act(exploit=True), rtol=0, atol=0
            )
        self.assertEqual(restored.policy_learner._training_steps, model._training_steps)
        # Noise is a new draw per update; match RNG state for the next-update
        # comparison promised by the model/optimizer checkpoint contract.
        batch = replay.sample(4)
        torch.manual_seed(999)
        agent.learn_batch(copy.deepcopy(batch))
        torch.manual_seed(999)
        restored.learn_batch(copy.deepcopy(batch))
        for a, b in zip(model._Q.parameters(), restored.policy_learner._Q.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
