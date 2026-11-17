# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Scalar mathematical oracles and PEARL lifecycle checks for TQC."""

import copy
import io
import math
import unittest
from unittest.mock import patch

import torch
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.quantile_critic import (
    QuantileCriticEnsemble,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.tqc import (
    TruncatedQuantileCritics,
    quantile_huber_loss,
    truncated_quantiles,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.risk_sensitive_safety_modules import RiskNeutralSafetyModule
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


def action_space():
    return BoxActionSpace(torch.tensor([-2.0, 1.0]), torch.tensor([4.0, 5.0]))


def learner(**kwargs):
    settings = dict(
        action_space=action_space(),
        state_dim=3,
        actor_hidden_dims=[8, 8],
        critic_hidden_dims=[8, 8],
        num_critics=2,
        num_quantiles=3,
        top_quantiles_to_drop_per_net=1,
        batch_size=4,
        entropy_coef=0.2,
        entropy_autotune=True,
    )
    settings.update(kwargs)
    return TruncatedQuantileCritics(**settings)


def batch(terminal=False):
    return TransitionBatch(
        state=torch.tensor([[1.0, 0.0, 0.25], [-0.2, 1.0, 0.5]]),
        action=torch.tensor([[0.0, 2.0], [1.0, 3.0]]),
        reward=torch.tensor([1.0, -0.5]),
        next_state=(
            None if terminal else torch.tensor([[0.2, 1.0, 0.0], [0.5, 0.0, 1.0]])
        ),
        terminated=torch.tensor([terminal, terminal]),
        truncated=torch.tensor([False, not terminal]),
    )


class TestTQCMathematics(unittest.TestCase):
    def test_pool_then_truncate_differs_from_per_critic_truncation(self):
        atoms = torch.tensor([[[0.0, 1.0, 2.0], [100.0, 101.0, 102.0]]])
        torch.testing.assert_close(
            truncated_quantiles(atoms, 1), torch.tensor([[0.0, 1.0, 2.0, 100.0]])
        )
        torch.testing.assert_close(
            truncated_quantiles(atoms, 0),
            torch.tensor([[0.0, 1.0, 2.0, 100.0, 101.0, 102.0]]),
        )
        for drop in [True, -1, 3, 1.5]:
            with self.subTest(drop=drop), self.assertRaises(ValueError):
                truncated_quantiles(atoms, drop)

    def test_quantile_loss_and_gradient_match_scalar_oracle(self):
        prediction = torch.tensor(
            [[[0.2, -0.7], [1.5, 0.1]], [[0.5, 1.2], [-2.0, 0.4]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor(
            [[-1.0, 0.4, 2.0], [0.2, 0.7, -0.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        tau = torch.tensor([0.25, 0.75], dtype=torch.float64)
        actual = quantile_huber_loss(prediction, target, tau)
        expected = torch.zeros(2, dtype=torch.float64)
        expected_gradient = torch.zeros_like(prediction)
        for b in range(2):
            for n in range(2):
                for m in range(2):
                    for k in range(3):
                        delta = float((target[b, k] - prediction[b, n, m]).detach())
                        weight = abs(float(tau[m]) - (1.0 if delta < 0 else 0.0))
                        huber = (
                            0.5 * delta * delta if abs(delta) <= 1 else abs(delta) - 0.5
                        )
                        expected[b] += weight * huber / 12
                        expected_gradient[b, n, m] += (
                            -weight * max(-1.0, min(1.0, delta)) / 24
                        )
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        torch.testing.assert_close(prediction.grad, expected_gradient)
        self.assertIsNone(target.grad)

    def test_terminal_targets_skip_nan_and_truncations_bootstrap(self):
        model = learner(discount_factor=0.5, entropy_autotune=False, entropy_coef=0.2)
        samples = batch()
        samples.terminated = torch.tensor([True, False])
        samples.next_state[0] = float("nan")
        with patch.object(
            model,
            "_policy_action_and_log_prob",
            return_value=(torch.zeros(1, 2), torch.tensor([[-2.0]])),
        ) as actor:
            with patch.object(
                model._critic_target,
                "get_q_value_distribution",
                return_value=torch.tensor([[[0.0, 1.0, 2.0], [100.0, 101.0, 102.0]]]),
            ):
                targets = model._target_quantiles(samples)
        torch.testing.assert_close(targets[0], torch.ones(4))
        torch.testing.assert_close(targets[1], torch.tensor([-0.3, 0.2, 0.7, 49.7]))
        self.assertEqual(actor.call_args.args[0].shape, (1, 3))
        self.assertTrue(torch.isfinite(actor.call_args.args[0]).all())
        terminal = batch(terminal=True)
        with patch.object(
            model._actor,
            "sample_action",
            side_effect=AssertionError("terminal actor evaluated"),
        ):
            torch.testing.assert_close(
                model._target_quantiles(terminal),
                terminal.reward[:, None].expand(-1, 4),
            )
        self.assertFalse(targets.requires_grad)

    def test_actor_uses_untruncated_mean_and_does_not_train_critics(self):
        model = learner(entropy_autotune=False, entropy_coef=0.0)
        actions = torch.zeros(2, 2, requires_grad=True)

        def critics(states, action):
            return (
                torch.tensor([[[0.0, 1.0, 2.0], [100.0, 101.0, 102.0]]] * 2)
                + action.sum(-1)[:, None, None]
            )

        with patch.object(
            model,
            "_policy_action_and_log_prob",
            return_value=(actions, torch.zeros(2, 1)),
        ):
            with patch.object(
                model._critic, "get_q_value_distribution", side_effect=critics
            ):
                loss = model._actor_loss(batch())
        self.assertAlmostEqual(loss.item(), -51.0)
        loss.backward()
        torch.testing.assert_close(actions.grad, torch.full((2, 2), -0.5))
        model._actor_loss(batch()).backward()
        self.assertTrue(any(p.grad is not None for p in model._actor.parameters()))
        self.assertTrue(all(p.grad is None for p in model._critic.parameters()))
        self.assertTrue(all(p.requires_grad for p in model._critic.parameters()))

    def test_normalized_entropy_density_matches_affine_jacobian(self):
        model = learner()
        states = torch.zeros(2, 3)
        actions = torch.zeros(2, 2)
        physical = torch.tensor([[-2.0], [-3.0]])
        with patch.object(
            model._actor, "sample_action", return_value=(actions, physical)
        ):
            _, normalized = model._policy_action_and_log_prob(states)
        torch.testing.assert_close(normalized, physical + math.log(6.0))
        model._entropy_in_normalized_coordinates = False
        with patch.object(
            model._actor, "sample_action", return_value=(actions, physical)
        ):
            _, unchanged = model._policy_action_and_log_prob(states)
        torch.testing.assert_close(unchanged, physical)

    def test_actual_gaussian_sampling_density_on_asymmetric_box(self):
        model = learner()
        with torch.no_grad():
            for parameter in model._actor.parameters():
                parameter.zero_()
        states = torch.zeros(2, 3)
        pre_tanh = torch.tensor([[0.2, -0.5], [1.0, -0.1]])
        with patch("torch.distributions.Normal.rsample", return_value=pre_tanh):
            actions, log_density = model._policy_action_and_log_prob(states)
        normalized = pre_tanh.tanh()
        scale = torch.tensor([3.0, 2.0])
        torch.testing.assert_close(
            actions, torch.tensor([1.0, 3.0]) + scale * normalized
        )
        # Native PEARL maps zero log-std logits to (-5+2)/2 = -1.5.
        normal_log_density = (
            -0.5 * (pre_tanh / math.exp(-1.5)).square()
            + 1.5
            - 0.5 * math.log(2 * math.pi)
        )
        physical_log_density = (
            normal_log_density - (scale * (1 - normalized.square()) + 1e-6).log()
        ).sum(1, keepdim=True)
        torch.testing.assert_close(
            log_density, physical_log_density + scale.log().sum()
        )

    def test_critic_gradient_does_not_train_actor_or_target(self):
        model = learner()
        model._critic_loss(batch()).backward()
        self.assertTrue(any(p.grad is not None for p in model._critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model._critic_target.parameters()))
        self.assertTrue(all(p.grad is None for p in model._actor.parameters()))


class TestTQCIntegration(unittest.TestCase):
    def test_cost_shaping_aligns_column_cost_with_flat_reward(self):
        model = learner()
        safety = torch.nn.Module()
        safety.register_buffer("lambda_constraint", torch.tensor(2.0))
        model.safety_module = safety
        samples = batch(terminal=True)
        samples.cost = torch.tensor([[0.2], [0.3]])
        prepared = model.preprocess_batch(samples)
        self.assertEqual(prepared.reward.shape, (2,))
        torch.testing.assert_close(prepared.reward, torch.tensor([0.6, -1.1]))

    def test_ensemble_members_are_independent_and_expected_q_is_full_mean(self):
        network = QuantileCriticEnsemble(3, 2, [8], num_critics=3, num_quantiles=7)
        states, actions = torch.rand(4, 3), torch.rand(4, 2)
        atoms = network.get_q_value_distribution(states, actions)
        self.assertEqual(atoms.shape, (4, 3, 7))
        torch.testing.assert_close(
            network.get_q_values(states, actions), atoms.mean((1, 2))
        )
        sets = [{id(p) for p in critic.parameters()} for critic in network.critics]
        self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2])

    def test_target_polyak_update_and_actual_counters(self):
        torch.manual_seed(25)
        model = learner(critic_soft_update_tau=0.1)
        before = [p.detach().clone() for p in model._critic_target.parameters()]
        report = model.learn_batch(batch())
        self.assertEqual(report["gradient_steps"], 1)
        self.assertEqual(model._gradient_steps, 1)
        for old, target, current in zip(
            before, model._critic_target.parameters(), model._critic.parameters()
        ):
            torch.testing.assert_close(target, 0.9 * old + 0.1 * current)
        self.assertTrue(
            all(not p.requires_grad for p in model._critic_target.parameters())
        )
        self.assertTrue(all(p.grad is None for p in model._critic_target.parameters()))
        self.assertGreater(model._entropy_coef.item(), 0.0)

    def test_deterministic_evaluation_respects_asymmetric_bounds(self):
        model = learner()
        state = torch.ones(3)
        with torch.no_grad():
            mean, _ = model._actor(state)
        expected = torch.tensor([1.0, 3.0]) + torch.tensor([3.0, 2.0]) * mean.tanh()
        torch.testing.assert_close(
            model.act(state, action_space(), exploit=True), expected
        )
        for _ in range(3):
            torch.testing.assert_close(
                model.act(state, action_space(), exploit=True), expected
            )
        self.assertTrue(
            (expected >= action_space().low).all()
            and (expected <= action_space().high).all()
        )

    def test_checkpoint_restores_populated_optimizers_and_next_update(self):
        torch.manual_seed(25)
        model = learner()
        model.learn_batch(batch())
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = learner()
        restored.load_state_dict(
            torch.load(stream, weights_only=True, map_location="cpu")
        )
        self.assertEqual(restored._gradient_steps, 1)
        self.assertTrue(restored._entropy_optimizer.state)
        torch.manual_seed(98)
        a = model.learn_batch(batch())
        torch.manual_seed(98)
        b = restored.learn_batch(batch())
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=6)
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q)
        with self.assertRaises(ValueError):
            learner(top_quantiles_to_drop_per_net=0).load_state_dict(model.state_dict())

    def test_terminal_lstm_history_is_trainable_and_owned_once(self):
        history = LSTMHistorySummarizationModule(
            observation_dim=3, action_dim=2, hidden_dim=3, history_length=2
        )
        model = learner()
        agent = PearlAgent(model, history_summarization_module=history)
        samples = batch(terminal=True)
        samples.state = torch.rand(2, 2, 5)
        samples = samples.to(agent.device)
        report = agent.learn_batch(samples)
        self.assertTrue(math.isfinite(report["critic_loss"]))
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in history.parameters()
            )
        )
        owned = [
            p
            for group in model._history_summarization_optimizer.param_groups
            for p in group["params"]
        ]
        for p in history.parameters():
            self.assertEqual(sum(q is p for q in owned), 1)
        other = [
            p
            for optimizer in [model._actor_optimizer, model._critic_optimizer]
            for group in optimizer.param_groups
            for p in group["params"]
        ]
        self.assertFalse({id(p) for p in owned} & {id(p) for p in other})

    def test_invalid_configs_and_risk_modules_are_rejected(self):
        for kwargs in [
            dict(num_critics=True),
            dict(num_quantiles=0),
            dict(top_quantiles_to_drop_per_net=3),
            dict(discount_factor=float("nan")),
            dict(critic_soft_update_tau=0),
            dict(entropy_coef=0),
            dict(batch_size=0),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                learner(**kwargs)
        model = learner()
        model.safety_module = RiskNeutralSafetyModule()
        with self.assertRaises(TypeError):
            model.act(torch.zeros(3), action_space())

    def test_real_pendulum_native_agent_replay_updates(self):
        torch.manual_seed(24)
        env = GymEnvironment("Pendulum-v1")
        try:
            model = learner(action_space=env.action_space, entropy_autotune=False)
            agent = PearlAgent(model, replay_buffer=BasicReplayBuffer(128))
            state, actions = env.reset(seed=24)
            agent.reset(state, actions)
            before = [p.detach().clone() for p in model._critic.parameters()]
            for step in range(48):
                result = env.step(agent.act())
                if step == 47:
                    result.truncated = True
                agent.observe(result)
                if step >= 8:
                    agent.learn()
                if result.done and step < 47:
                    state, actions = env.reset(seed=24 + step)
                    agent.reset(state, actions)
            self.assertEqual(len(agent.replay_buffer), 48)
            self.assertEqual(model._training_steps, 40)
            self.assertEqual(model._gradient_steps, 40)
            self.assertTrue(
                any(
                    not torch.equal(p, q)
                    for p, q in zip(model._critic.parameters(), before)
                )
            )
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA device")
    def test_populated_checkpoint_cpu_cuda_cpu(self):
        torch.manual_seed(9)
        cpu = learner()
        cpu.learn_batch(batch())
        gpu = learner().cuda()
        gpu.load_state_dict(copy.deepcopy(cpu.state_dict()))
        report = gpu.learn_batch(batch().to(torch.device("cuda")))
        self.assertTrue(math.isfinite(report["critic_loss"]))
        for optimizer in [
            gpu._actor_optimizer,
            gpu._critic_optimizer,
            gpu._entropy_optimizer,
        ]:
            for parameter, state in optimizer.state.items():
                self.assertEqual(state["exp_avg"].device, parameter.device)
        restored = learner()
        restored.load_state_dict(copy.deepcopy(gpu.state_dict()))
        self.assertEqual(restored._gradient_steps, 2)
        for p, q in zip(restored.parameters(), gpu.parameters()):
            torch.testing.assert_close(p, q.cpu())
        self.assertTrue(math.isfinite(restored.learn_batch(batch())["actor_loss"]))


if __name__ == "__main__":
    unittest.main()
