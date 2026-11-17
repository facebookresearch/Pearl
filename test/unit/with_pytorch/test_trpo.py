# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import unittest
from unittest import mock

import torch
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.continuous_ppo import (
    ContinuousPPORolloutBuffer,
    ContinuousPPOTransitionBatch,
)
from pearl.policy_learners.sequential_decision_making.trpo import (
    ContinuousTrustRegionPolicyOptimization,
    conjugate_gradient,
    gaussian_kl,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.safety_modules.risk_sensitive_safety_modules import RiskNeutralSafetyModule
from pearl.utils.functional_utils.train_and_eval.continuous_ppo_vector_collector import (
    ContinuousPPOVectorCollector,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from torch.distributions import Normal, kl_divergence


def action_space(dim=2):
    return BoxActionSpace(torch.full((dim,), -0.2), torch.full((dim,), 0.4))


def policy(**kwargs):
    options = dict(
        state_dim=3,
        action_space=action_space(),
        hidden_dims=[8],
        critic_update_epochs=2,
        batch_size=32,
    )
    options.update(kwargs)
    return ContinuousTrustRegionPolicyOptimization(**options)


def actor_vector(learner):
    return torch.cat(
        [p.detach().reshape(-1) for p in learner._actor.parameters()]
    ).clone()


def rollout(learner, count=128, sampling_batch=None):
    parameter = next(learner.parameters())
    states = torch.randn(
        count, learner._state_dim, device=parameter.device, dtype=parameter.dtype
    )
    parts = [
        learner.sample_actions(states[i : i + (sampling_batch or count)])
        for i in range(0, count, sampling_batch or count)
    ]
    actions, latent, log_prob, values = [
        torch.cat([part[i] for part in parts]) for i in range(4)
    ]
    return ContinuousPPOTransitionBatch(
        state=states,
        action=actions,
        latent_action=latent,
        old_log_prob=log_prob,
        old_value=values,
        reward=torch.ones_like(values),
        terminated=torch.zeros(count, dtype=torch.bool, device=states.device),
        truncated=torch.zeros(count, dtype=torch.bool, device=states.device),
        advantages=latent[:, 0] + 0.3 * states[:, 0],
        returns=values + 1,
        behavior_policy_version=learner._policy_version,
    )


class BoundaryEnvironment(Environment):
    def __init__(self, terminal=False):
        self.terminal = terminal

    @property
    def action_space(self):
        return action_space(1)

    @property
    def observation_space(self):
        return BoxActionSpace(torch.tensor([-1000.0]), torch.tensor([1000.0]))

    def reset(self, seed=None):
        return torch.tensor([1.0]), self.action_space

    def step(self, action):
        return ActionResult(
            observation=None if self.terminal else torch.tensor([999.0]),
            reward=2.0,
            terminated=self.terminal,
            truncated=not self.terminal,
            info={} if self.terminal else {"final_observation": torch.tensor([3.0])},
        )


class TestTRPO(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_joint_old_to_new_kl_is_asymmetric_and_sums_actions(self):
        old_mean = torch.tensor([[0.0, 1.0], [-0.5, 0.3]], dtype=torch.float64)
        old_std = torch.tensor([[1.0, 2.0], [0.7, 1.3]], dtype=torch.float64)
        new_mean, new_std = old_mean + 0.4, old_std * torch.tensor([1.8, 0.6])
        expected = (
            kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std))
            .sum(-1)
            .mean()
        )
        actual = gaussian_kl(old_mean, old_std, new_mean, new_std)
        torch.testing.assert_close(actual, expected)
        self.assertNotAlmostEqual(
            float(actual), float(gaussian_kl(new_mean, new_std, old_mean, old_std))
        )
        self.assertNotAlmostEqual(float(actual), float(expected / 2))

    def test_conjugate_gradient_matches_dense_solve_and_reports_breakdown(self):
        matrix = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 0.2], [0.0, 0.2, 2.0]], dtype=torch.float64
        )
        rhs = torch.tensor([1.0, -2.0, 0.3], dtype=torch.float64)
        result = conjugate_gradient(lambda value: matrix @ value, rhs, 10, 1e-12)
        torch.testing.assert_close(
            result.solution, torch.linalg.solve(matrix, rhs), atol=1e-12, rtol=1e-12
        )
        self.assertTrue(result.converged)
        self.assertFalse(result.breakdown)
        self.assertTrue(conjugate_gradient(lambda value: -value, rhs).breakdown)

    def test_fisher_product_matches_independent_dense_hessian_including_logstd(self):
        learner = policy(state_dim=1, hidden_dims=[]).double()
        states = torch.tensor([[-0.7], [0.2], [1.1]], dtype=torch.float64)
        actor = learner._actor
        initial = actor_vector(learner).requires_grad_(True)
        with torch.no_grad():
            old = actor.distribution(states)
            old_mean, old_std = old.mean.clone(), old.stddev.clone()
        names = [
            (name, value.shape, value.numel())
            for name, value in actor.named_parameters()
        ]

        def independent_kl(flat):
            parameters, offset = {}, 0
            for name, shape, count in names:
                parameters[name] = flat[offset : offset + count].view(shape)
                offset += count
            mean = torch.func.functional_call(actor, parameters, (states,))
            std = parameters["log_std"].exp().expand_as(mean)
            return (
                kl_divergence(Normal(old_mean, old_std), Normal(mean, std))
                .sum(-1)
                .mean()
            )

        hessian = torch.autograd.functional.hessian(independent_kl, initial)
        vector = torch.linspace(-0.5, 0.8, initial.numel(), dtype=torch.float64)
        actual = learner.fisher_vector_product(
            states, old_mean, old_std, vector, damping=0.2
        )
        torch.testing.assert_close(
            actual, hessian @ vector + 0.2 * vector, atol=1e-11, rtol=1e-11
        )
        self.assertTrue(all(p.grad is None for p in learner._critic.parameters()))

    def test_surrogate_gradient_matches_gaussian_score_oracle(self):
        learner = policy(
            state_dim=1,
            action_space=action_space(1),
            hidden_dims=[],
            normalize_advantages=False,
        ).double()
        with torch.no_grad():
            for p in learner._actor.parameters():
                p.zero_()
        states = torch.tensor([[-1.0], [0.2], [0.5], [1.3]], dtype=torch.float64)
        latent = torch.tensor([[-0.5], [0.2], [1.0], [1.3]], dtype=torch.float64)
        advantage = torch.tensor([-1.0, 0.5, 0.8, 1.7], dtype=torch.float64)
        batch = ContinuousPPOTransitionBatch(
            state=states,
            action=latent.clamp(-0.2, 0.4),
            latent_action=latent,
            reward=torch.zeros(4),
            terminated=torch.zeros(4, dtype=torch.bool),
            old_log_prob=Normal(torch.zeros_like(latent), torch.ones_like(latent))
            .log_prob(latent)
            .sum(-1),
            old_value=torch.zeros(4),
            advantages=advantage,
            returns=torch.zeros(4),
            behavior_policy_version=0,
        )
        parameters = dict(learner._actor.named_parameters())
        gradients = torch.autograd.grad(
            -learner._actor_loss(batch), list(parameters.values())
        )
        actual = dict(zip(parameters, gradients))
        torch.testing.assert_close(
            actual["log_std"][0], (advantage * (latent[:, 0].square() - 1)).mean()
        )
        torch.testing.assert_close(
            actual["mean_network.0.weight"][0, 0],
            (advantage * latent[:, 0] * states[:, 0]).mean(),
        )
        torch.testing.assert_close(
            actual["mean_network.0.bias"][0], (advantage * latent[:, 0]).mean()
        )

    def test_accepted_step_improves_surrogate_obeys_real_kl_and_isolates_critic(self):
        learner = policy()
        batch = rollout(learner)
        old = learner._actor.distribution(batch.state)
        old_mean, old_std = old.mean.detach().clone(), old.stddev.detach().clone()
        initial = actor_vector(learner)
        actors_during_critic = []
        hook = learner._critic_optimizer.register_step_pre_hook(
            lambda *args: actors_during_critic.append(actor_vector(learner))
        )
        try:
            report = learner.learn_rollout(batch)
        finally:
            hook.remove()
        self.assertEqual(report["actor_step_accepted"], [1.0])
        self.assertIsNone(learner._actor_optimizer)
        self.assertFalse(torch.equal(initial, actor_vector(learner)))
        self.assertTrue(
            all(
                torch.equal(value, actor_vector(learner))
                for value in actors_during_critic
            )
        )
        current = learner._actor.distribution(batch.state)
        measured = kl_divergence(Normal(old_mean, old_std), current).sum(-1).mean()
        self.assertLessEqual(float(measured.detach()), learner._target_kl)
        self.assertAlmostEqual(
            report["actor_kl"][0], float(measured.detach()), places=6
        )
        self.assertGreater(report["surrogate_improvement"][0], 0)
        self.assertTrue(all(p.grad is None for p in learner._actor.parameters()))

    def test_rejected_finite_search_restores_mutated_actor_and_consumes_rollout(self):
        learner = policy(line_search_max_iterations=3)
        batch = rollout(learner)
        original, applied = actor_vector(learner), []
        setter = learner._set_actor_parameters

        def record(flat):
            applied.append(flat.detach().clone())
            setter(flat)

        with mock.patch.object(
            learner, "_set_actor_parameters", side_effect=record
        ), mock.patch.object(learner, "_accept_actor_candidate", return_value=False):
            result = learner.learn_rollout(batch)
        self.assertTrue(any(not torch.equal(flat, original) for flat in applied))
        self.assertTrue(torch.equal(actor_vector(learner), original))
        self.assertEqual(result["actor_step_accepted"], [0.0])
        self.assertEqual(result["actor_numerical_failure"], [0.0])
        with self.assertRaisesRegex(ValueError, "fresh"):
            learner.learn_rollout(batch)

    def test_nonfinite_candidates_restore_actor_and_are_reported_separately(self):
        learner = policy(line_search_max_iterations=3)
        batch, original = rollout(learner), actor_vector(learner)
        distribution = learner._actor.distribution

        def damaged(states, *args):
            if not torch.equal(actor_vector(learner), original):
                return Normal(
                    torch.full((len(states), 2), float("nan"), device=states.device),
                    torch.ones(len(states), 2, device=states.device),
                    validate_args=False,
                )
            return distribution(states, *args)

        with mock.patch.object(learner._actor, "distribution", side_effect=damaged):
            result = learner.learn_rollout(batch)
        self.assertTrue(torch.equal(actor_vector(learner), original))
        self.assertEqual(result["actor_step_accepted"], [0.0])
        self.assertEqual(result["actor_numerical_failure"], [1.0])
        self.assertEqual(result["line_search_nonfinite_candidates"], [3.0])

    def test_exception_during_search_restores_actor(self):
        learner = policy()
        batch, original = rollout(learner), actor_vector(learner)
        with mock.patch.object(
            learner,
            "_accept_actor_candidate",
            side_effect=RuntimeError("search failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "search failed"):
                learner.learn_rollout(batch)
        self.assertTrue(torch.equal(actor_vector(learner), original))

    def test_zero_gradient_is_finite_rejection_and_critic_still_learns(self):
        learner = policy()
        batch = rollout(learner)
        batch.advantages.zero_()
        original = actor_vector(learner)
        critic = [p.detach().clone() for p in learner._critic.parameters()]
        result = learner.learn_rollout(batch)
        self.assertTrue(torch.equal(original, actor_vector(learner)))
        self.assertEqual(result["actor_numerical_failure"], [0.0])
        self.assertEqual(
            learner.actor_update_statistics["last_rejection_reason"], "zero_gradient"
        )
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(critic, learner._critic.parameters())
            )
        )

    def test_collection_batch_roundoff_is_allowed_but_stale_data_is_rejected(self):
        learner = policy()
        batch = rollout(learner, sampling_batch=4)
        batch.old_log_prob += 5e-6
        learner.learn_rollout(batch)
        with self.assertRaisesRegex(ValueError, "fresh"):
            learner.learn_rollout(batch)
        batch = rollout(learner)
        batch.old_log_prob += 0.01
        original = actor_vector(learner)
        with self.assertRaisesRegex(ValueError, "likelihoods"):
            learner.learn_rollout(batch)
        self.assertTrue(torch.equal(original, actor_vector(learner)))

    def test_final_observations_bootstrap_truncation_without_cross_reset_trace(self):
        learner = policy(
            state_dim=1,
            action_space=action_space(1),
            hidden_dims=[],
            discount_factor=0.9,
        )
        with torch.no_grad():
            learner._critic.model[0].weight.fill_(1)
            learner._critic.model[0].bias.zero_()
        collector = ContinuousPPOVectorCollector(
            [BoundaryEnvironment(), BoundaryEnvironment(True)],
            learner,
            31,
            normalize_observations=False,
            normalize_rewards=False,
        )
        try:
            batch = collector.collect(2)
            torch.testing.assert_close(
                batch.advantages, torch.tensor([3.7, 1.0, 3.7, 1.0])
            )
            torch.testing.assert_close(
                batch.returns, torch.tensor([4.7, 2.0, 4.7, 2.0])
            )
        finally:
            collector.close()

    def test_native_serial_agent_uses_behavior_rollout_and_rejects_generic_replay(self):
        learner = policy(state_dim=1, action_space=action_space(1), hidden_dims=[])
        buffer = ContinuousPPORolloutBuffer(8, learner)
        agent = PearlAgent(learner, replay_buffer=buffer, device_id=-1)
        env = BoundaryEnvironment()
        state, actions = env.reset()
        agent.reset(state, actions)
        for _ in range(4):
            result = env.step(agent.act())
            result.observation = result.info["final_observation"]
            agent.observe(result)
            agent.reset(*env.reset())
        report = agent.learn()
        self.assertEqual(report["actor_update_attempted"], [1.0])
        self.assertEqual(len(buffer), 0)
        with self.assertRaises(TypeError):
            learner.learn(BasicReplayBuffer(8))

    def test_native_vector_pendulum_runs_collection_and_value_updates(self):
        environments = [GymEnvironment("Pendulum-v1") for _ in range(4)]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        learner = policy(state_dim=3, action_space=environments[0].action_space).to(
            device
        )
        collector = ContinuousPPOVectorCollector(environments, learner, 61)
        try:
            batch = collector.collect(64)
            result = learner.learn_rollout(batch)
            self.assertEqual(collector.steps, 256)
            self.assertEqual(
                result["actor_step_accepted"][0] + result["actor_step_rejected"][0], 1.0
            )
            self.assertEqual(result["actor_numerical_failure"], [0.0])
            self.assertTrue(learner._critic_optimizer.state)
            self.assertLessEqual(result["actor_kl"][0], learner._target_kl)
        finally:
            collector.close()

    def test_transition_device_conversion_preserves_integer_policy_version(self):
        learner = policy()
        batch = rollout(learner).to(torch.device("cpu"))
        self.assertIsInstance(batch.behavior_policy_version, torch.Tensor)
        learner.learn_rollout(batch)

    def test_checkpoint_restores_populated_critic_optimizer_and_actor_counters(self):
        learner = policy()
        learner.learn_rollout(rollout(learner))
        stream = io.BytesIO()
        torch.save(learner.state_dict(), stream)
        stream.seek(0)
        restored = policy()
        restored.load_state_dict(
            torch.load(stream, map_location="cpu", weights_only=True)
        )
        self.assertEqual(
            learner.actor_update_statistics, restored.actor_update_statistics
        )
        self.assertEqual(learner.compare(restored), "")
        self.assertTrue(restored._critic_optimizer.state)
        self.assertTrue(torch.equal(actor_vector(learner), actor_vector(restored)))
        next(iter(restored._critic_optimizer.state.values()))["exp_avg"].add_(0.1)
        self.assertIn("optimizer differs", learner.compare(restored))
        restored.load_state_dict(copy.deepcopy(learner.state_dict()))
        restored.learn_rollout(rollout(restored))

    def test_bounds_components_and_invalid_versions_are_rejected(self):
        learner = policy()
        with self.assertRaises(ValueError):
            learner.reset(
                BoxActionSpace(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
            )
        learner.safety_module = RiskNeutralSafetyModule()
        with self.assertRaises(TypeError):
            learner.sample_actions(torch.ones(2, 3))
        learner = policy()
        batch = rollout(learner)
        batch.behavior_policy_version = 0.5
        with self.assertRaises(ValueError):
            learner.learn_rollout(batch)

    def test_shape_retry_uses_fresh_storage_and_preserves_stale_checks(self):
        learner = policy()
        batch = rollout(learner, sampling_batch=4)
        learner.sample_actions(torch.zeros(1, 3), exploit=True)
        self.assertEqual(learner._behavior_sampling_batch_size, 4)
        original = learner._actor.get_action_log_prob
        calls = []

        def different_full_batch(states, actions):
            calls.append(
                (len(states), states.storage_offset(), actions.storage_offset())
            )
            values = original(states, actions)
            return values + 0.01 if len(states) == len(batch) else values

        with mock.patch.object(
            learner._actor, "get_action_log_prob", side_effect=different_full_batch
        ):
            learner._validate_rollout(batch)
            self.assertTrue(any(size == 4 for size, _, _ in calls))
            self.assertTrue(
                all(state == action == 0 for size, state, action in calls if size == 4)
            )
            calls.clear()
            batch.behavior_policy_version += 1
            with self.assertRaisesRegex(ValueError, "fresh"):
                learner._validate_rollout(batch)
            self.assertEqual(calls, [])
            batch.behavior_policy_version -= 1
            bias = learner._actor.mean_network[-1].bias
            saved = bias.detach().clone()
            with torch.no_grad():
                bias.add_(0.01)
            with self.assertRaisesRegex(ValueError, "likelihoods"):
                learner._validate_rollout(batch)
            self.assertTrue(any(size == 4 for size, _, _ in calls))
            with torch.no_grad():
                bias.copy_(saved)

    def test_mixed_sampling_sizes_and_checkpoint_load_disable_shape_retry(self):
        learner = policy()
        batch = rollout(learner, sampling_batch=4)
        original = learner._actor.get_action_log_prob

        def different_full_batch(states, actions):
            values = original(states, actions)
            return values + 0.01 if len(states) == len(batch) else values

        learner.sample_actions(torch.zeros(1, 3))
        learner.sample_actions(torch.zeros(4, 3))
        self.assertIsNone(learner._behavior_sampling_batch_size)
        with mock.patch.object(
            learner._actor, "get_action_log_prob", side_effect=different_full_batch
        ):
            with self.assertRaisesRegex(ValueError, "likelihoods"):
                learner._validate_rollout(batch)
        learner.load_state_dict(copy.deepcopy(learner.state_dict()))
        self.assertIsNone(learner._behavior_sampling_batch_size)
        self.assertIsNone(learner._behavior_sampling_policy_version)
        with mock.patch.object(
            learner._actor, "get_action_log_prob", side_effect=different_full_batch
        ):
            with self.assertRaisesRegex(ValueError, "likelihoods"):
                learner._validate_rollout(batch)
            fresh = rollout(learner, sampling_batch=4)
            fresh.advantages.zero_()
            result = learner.learn_rollout(fresh)
        self.assertEqual(result["actor_step_accepted"], [0.0])
        self.assertIsNone(learner._behavior_sampling_batch_size)
        self.assertIsNone(learner._behavior_sampling_policy_version)

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA")
    def test_cuda_sharp_policy_accepts_fresh_small_batch_rollout(self):
        for log_std in [-3.0, -4.0, -5.0]:
            with self.subTest(log_std=log_std):
                torch.manual_seed(868)
                learner = ContinuousTrustRegionPolicyOptimization(
                    105,
                    BoxActionSpace(-torch.ones(8), torch.ones(8)),
                    critic_update_epochs=1,
                ).cuda()
                with torch.no_grad():
                    learner._actor.mean_network[-1].weight.mul_(100)
                    learner._actor.log_std.fill_(log_std)
                states, samples = [], []
                for chunk in torch.randn(2048, 105).split(4):
                    state = chunk.clone().cuda()
                    states.append(state)
                    samples.append(learner.sample_actions(state))
                action, latent, log_prob, values = [
                    torch.cat([sample[i] for sample in samples]) for i in range(4)
                ]
                state = torch.cat(states)
                self.assertGreater(state[4:8].storage_offset(), 0)
                batch = ContinuousPPOTransitionBatch(
                    state=state,
                    action=action,
                    latent_action=latent,
                    reward=torch.ones_like(values),
                    terminated=torch.zeros(2048, dtype=torch.bool, device="cuda"),
                    truncated=torch.zeros(2048, dtype=torch.bool, device="cuda"),
                    old_log_prob=log_prob,
                    old_value=values,
                    advantages=latent[:, 0] + 0.3 * state[:, 0],
                    returns=values + 1,
                    behavior_policy_version=0,
                )
                learner.sample_actions(torch.zeros(1, 105, device="cuda"), exploit=True)
                learner._validate_rollout(batch)
                batch.behavior_policy_version = 1
                with self.assertRaisesRegex(ValueError, "fresh"):
                    learner._validate_rollout(batch)
                batch.behavior_policy_version = 0
                bias = learner._actor.mean_network[-1].bias
                saved = bias.detach().clone()
                with torch.no_grad():
                    bias.add_(0.01)
                with self.assertRaisesRegex(ValueError, "likelihoods"):
                    learner._validate_rollout(batch)
                with torch.no_grad():
                    bias.copy_(saved)
                report = learner.learn_rollout(batch)
                self.assertEqual(report["actor_update_attempted"], [1.0])
                self.assertEqual(report["actor_numerical_failure"], [0.0])
                self.assertEqual(learner._policy_version, 1)
                self.assertTrue(learner._critic_optimizer.state)
                self.assertIsNone(learner._behavior_sampling_batch_size)

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA")
    def test_cuda_real_update_checkpoint_and_populated_optimizer_movement(self):
        learner = policy().cuda()
        batch = rollout(learner, sampling_batch=4)
        report = learner.learn_rollout(batch)
        self.assertEqual(report["actor_step_accepted"], [1.0])
        saved = copy.deepcopy(learner.state_dict())
        restored = policy()
        restored.load_state_dict(saved)
        self.assertEqual(learner.compare(restored), "")
        restored.cuda()
        self.assertTrue(
            all(
                value.device.type == "cuda"
                for state in restored._critic_optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        )
        torch.testing.assert_close(actor_vector(learner), actor_vector(restored))
        self.assertEqual(
            learner.actor_update_statistics, restored.actor_update_statistics
        )
        restored.learn_rollout(rollout(restored))
        restored.cpu()
        self.assertTrue(
            all(
                value.device.type == "cpu"
                for state in restored._critic_optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        )


if __name__ == "__main__":
    unittest.main()
