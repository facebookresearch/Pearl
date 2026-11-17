# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Analytic PPO/RPO objectives, behavior metadata, boundary and device contracts."""

import copy
import io
import json
import math
import os
import unittest

import numpy as np
import torch
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.gaussian_ppo_actor import (
    GaussianPPOActorNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.continuous_ppo import (
    ContinuousProximalPolicyOptimization,
    ContinuousPPORolloutBuffer,
    ContinuousPPOTransitionBatch,
    generalized_advantage_estimate,
)
from pearl.policy_learners.sequential_decision_making.robust_policy_optimization import (
    RobustPolicyOptimization,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.functional_utils.train_and_eval.continuous_ppo_vector_collector import (
    ContinuousPPOVectorCollector,
    RunningMoments,
    normalize_from_snapshot,
)


def space():
    return BoxActionSpace(torch.tensor([-1.0]), torch.tensor([1.0]))


def policy(kind=ContinuousProximalPolicyOptimization, **kwargs):
    options = dict(
        state_dim=2,
        action_space=space(),
        hidden_dims=[],
        batch_size=2,
        update_epochs=2,
        normalize_advantages=False,
    )
    options.update(kwargs)
    return kind(**options)


def batch():
    latent = torch.tensor([[1.0], [2.0], [-1.0], [0.5]])
    logp = -0.5 * latent[:, 0].square() - 0.5 * math.log(2 * math.pi)
    return ContinuousPPOTransitionBatch(
        state=torch.zeros(4, 2),
        action=latent.clamp(-1, 1),
        reward=torch.zeros(4),
        terminated=torch.ones(4, dtype=torch.bool),
        latent_action=latent,
        old_log_prob=logp - torch.tensor([1.5, 0.5, 1.1, 0.9]).log(),
        old_value=torch.zeros(4),
        advantages=torch.tensor([2.0, -1.0, 2.0, -1.0]),
        returns=torch.zeros(4),
    )


class TestContinuousPPO(unittest.TestCase):
    def test_normalized_policy_checkpoint_reloads_deterministic_actions(self):
        envs = [GymEnvironment("Pendulum-v1") for _ in range(2)]
        learner = ContinuousProximalPolicyOptimization(
            3, envs[0].action_space, hidden_dims=[8], batch_size=8, update_epochs=1
        )
        collector = ContinuousPPOVectorCollector(envs, learner, 71)
        learner.learn_rollout(collector.collect(8))
        snapshot = json.loads(json.dumps(collector.snapshot()))
        stream = io.BytesIO()
        torch.save(learner.state_dict(), stream)
        stream.seek(0)
        restored = ContinuousProximalPolicyOptimization(
            3, envs[0].action_space, hidden_dims=[8], batch_size=8, update_epochs=1
        )
        restored.load_state_dict(
            torch.load(stream, weights_only=True, map_location="cpu")
        )
        raw = torch.tensor([[1.0, 0.2, -0.4], [0.1, -0.9, 2.0]])
        expected = collector.normalize(raw)
        actual = normalize_from_snapshot(raw, snapshot)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            learner.sample_actions(expected, exploit=True)[0],
            restored.sample_actions(actual, exploit=True)[0],
        )
        collector.close()

    def test_compare_includes_algorithm_settings(self):
        left = policy()
        right = policy(clip_epsilon=0.1)
        self.assertIn("configuration differs", left.compare(right))
        self.assertIn("types differ", left.compare(policy(RobustPolicyOptimization)))

    def test_running_moments_matches_population_statistics(self):
        moments = RunningMoments((2,))
        values = torch.tensor([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0], [7.0, 16.0]])
        moments.update(values[:2])
        moments.update(values[2:])
        torch.testing.assert_close(
            moments.mean, values.double().mean(0), atol=0.001, rtol=0.0001
        )
        torch.testing.assert_close(
            moments.variance,
            values.double().var(0, unbiased=False),
            atol=0.01,
            rtol=0.001,
        )

    def test_vector_autoreset_bootstraps_final_observation_and_counts_transitions(self):
        class TimeoutEnvironment(Environment):
            def __init__(self, terminal):
                self.terminal = terminal

            @property
            def action_space(self):
                return space()

            @property
            def observation_space(self):
                return None

            def reset(self, seed=None):
                return torch.tensor([100.0, 0.0]), self.action_space

            def step(self, action):
                return ActionResult(
                    torch.tensor([999.0, 0.0]),
                    1.0,
                    self.terminal,
                    not self.terminal,
                    info={"final_observation": torch.tensor([3.0, 0.0])},
                )

        learner = policy()
        with torch.no_grad():
            learner._critic.model[-1].weight.copy_(torch.tensor([[2.0, 0.0]]))
            learner._critic.model[-1].bias.zero_()
        collector = ContinuousPPOVectorCollector(
            [TimeoutEnvironment(False), TimeoutEnvironment(True)],
            learner,
            17,
            normalize_observations=False,
            normalize_rewards=False,
        )
        data = collector.collect(2)
        torch.testing.assert_close(data.returns, torch.tensor([6.94, 1.0, 6.94, 1.0]))
        self.assertEqual(collector.steps, 4)
        self.assertEqual(len(collector.episodes), 4)
        self.assertEqual(data.behavior_policy_version, learner._policy_version)
        learner.learn_rollout(data)
        with self.assertRaises(ValueError):
            learner.learn_rollout(data)

    def test_real_vector_normalized_pendulum_for_both_methods(self):
        torch.set_num_threads(1)
        for kind in [ContinuousProximalPolicyOptimization, RobustPolicyOptimization]:
            envs = [GymEnvironment("Pendulum-v1") for _ in range(4)]
            learner = kind(
                3, envs[0].action_space, hidden_dims=[8], batch_size=32, update_epochs=2
            ).to(torch.device(os.environ.get("PEARL_TEST_DEVICE", "cpu")))
            collector = ContinuousPPOVectorCollector(envs, learner, 28)
            data = collector.collect(32)
            self.assertEqual(len(data), 128)
            report = learner.learn_rollout(data)
            self.assertEqual(len(report["loss"]), 8)
            snapshot = collector.snapshot()
            self.assertEqual(snapshot["steps"], 128)
            count = snapshot["observation_moments"]["count"]
            collector.normalize(torch.ones(4, 3))
            self.assertEqual(collector.observation_moments.count, count)
            self.assertTrue(all(math.isfinite(x) for x in report["loss"]))
            collector.close()

    def test_exploit_invalidates_stale_saturated_behavior_metadata(self):
        learner = policy()
        with torch.no_grad():
            learner._actor.mean_network[-1].weight.zero_()
            learner._actor.mean_network[-1].bias.fill_(5.0)
            learner._actor.log_std.fill_(-10.0)
        state = torch.zeros(2)
        stochastic = learner.act(state, space())
        evaluation = learner.act(state, space(), exploit=True)
        self.assertTrue(torch.equal(stochastic, evaluation))
        buffer = ContinuousPPORolloutBuffer(8, learner)
        with self.assertRaises(RuntimeError):
            buffer.push(state, evaluation, 1.0, False, False, next_state=state)

    def test_gae_matches_independent_finite_delta_sum_across_boundaries(self):
        generator = torch.Generator().manual_seed(29)
        reward = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        values = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        next_values = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        terminated = torch.zeros(8, 3, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        terminated[2, 0] = True
        truncated[3, 1] = True
        terminated[7, 2] = True
        expected = torch.zeros_like(values)
        for t in range(8):
            for env in range(3):
                for future in range(t, 8):
                    delta = (
                        reward[future, env]
                        + (
                            0.9 * next_values[future, env]
                            if not terminated[future, env]
                            else 0
                        )
                        - values[future, env]
                    )
                    expected[t, env] += (0.9 * 0.8) ** (future - t) * delta
                    if terminated[future, env] or truncated[future, env]:
                        break
        actual, returns = generalized_advantage_estimate(
            reward, values, next_values, terminated, truncated, 0.9, 0.8
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(returns, expected + values)
        self.assertFalse(actual.requires_grad)

    def test_truncation_bootstraps_final_state_and_stops_trace(self):
        args = [
            torch.tensor([[1.0], [100.0]]),
            torch.zeros(2, 1),
            torch.tensor([[7.0], [999.0]]),
        ]
        term = torch.tensor([[False], [True]])
        trunc = torch.tensor([[True], [False]])
        advantage, _ = generalized_advantage_estimate(*args, term, trunc, 0.9, 1.0)
        self.assertAlmostEqual(float(advantage[0]), 7.3, places=5)
        self.assertEqual(float(advantage[1]), 100.0)
        term[0] = True
        advantage, _ = generalized_advantage_estimate(*args, term, trunc, 0.9, 1.0)
        self.assertEqual(float(advantage[0]), 1.0)

    def test_clipped_policy_loss_and_analytic_gaussian_gradients(self):
        learner = policy()
        with torch.no_grad():
            for p in learner._actor.parameters():
                p.zero_()
        data = batch()
        data.old_log_prob.requires_grad_()
        data.advantages.requires_grad_()
        loss = learner._actor_loss(data)
        self.assertAlmostEqual(float(loss.detach()), -0.725, places=6)
        loss.backward()
        torch.testing.assert_close(
            learner._actor.mean_network[-1].bias.grad, torch.tensor([0.6625])
        )
        torch.testing.assert_close(
            learner._actor.log_std.grad, torch.tensor([-0.16875])
        )
        self.assertIsNone(data.old_log_prob.grad)
        self.assertIsNone(data.advantages.grad)

    def test_value_clipping_uses_maximum_squared_error(self):
        learner = policy()
        with torch.no_grad():
            for p in learner._critic.parameters():
                p.zero_()
            learner._critic.model[-1].bias.fill_(1.0)
        data = batch()
        data.returns.fill_(1.0)
        self.assertAlmostEqual(
            float(learner._critic_loss(data).detach()), 0.32, places=6
        )
        learner._clip_value_loss = False
        self.assertEqual(float(learner._critic_loss(data).detach()), 0.0)

    def test_rpo_zero_equals_ppo_and_perturbs_only_update_distribution(self):
        baseline = policy()
        rpo = policy(RobustPolicyOptimization, rpo_alpha=0.0)
        rpo._actor.load_state_dict(baseline._actor.state_dict())
        rpo._critic.load_state_dict(baseline._critic.state_dict())
        data = batch()
        torch.testing.assert_close(baseline._actor_loss(data), rpo._actor_loss(data))
        rpo._rpo_alpha = 0.5
        states = torch.zeros(100, 2)
        mean = rpo._actor(states)
        torch.manual_seed(44)
        perturbed = rpo._actor.distribution(states, 0.5).mean - mean
        self.assertTrue((perturbed.abs() <= 0.5).all())
        self.assertGreater(float(perturbed.abs().sum()), 1.0)
        torch.manual_seed(90)
        a = baseline.sample_actions(states)
        torch.manual_seed(90)
        b = rpo.sample_actions(states)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)

    def test_likelihood_uses_latent_sample_not_clipped_action(self):
        actor = GaussianPPOActorNetwork(2, [], 1, space())
        with torch.no_grad():
            actor.mean_network[-1].weight.zero_()
            actor.mean_network[-1].bias.fill_(5.0)
            actor.log_std.fill_(-3.0)
        latent = torch.tensor([[5.0]])
        self.assertEqual(float(actor.clip_action(latent)), 1.0)
        expected = 3.0 - 0.5 * math.log(2 * math.pi)
        self.assertAlmostEqual(
            float(actor.get_action_log_prob(torch.zeros(1, 2), latent).detach()),
            expected,
            places=5,
        )
        self.assertLess(
            float(
                actor.get_action_log_prob(
                    torch.zeros(1, 2), actor.clip_action(latent)
                ).detach()
            ),
            -100.0,
        )

    def test_native_agent_metadata_and_on_policy_clear(self):
        learner = policy()
        buffer = ContinuousPPORolloutBuffer(8, learner)
        agent = PearlAgent(learner, replay_buffer=buffer)
        agent.reset(torch.zeros(2), space())
        for i in range(8):
            action = agent.act()
            self.assertTrue((action >= -1).all() and (action <= 1).all())
            agent.observe(
                ActionResult(torch.tensor([float(i) / 10, 0.0]), 1.0, i == 7, False)
            )
        original = buffer.full_batch().old_log_prob.clone()
        with torch.no_grad():
            learner._actor.mean_network[-1].bias.add_(0.2)
        torch.testing.assert_close(buffer.full_batch().old_log_prob, original)
        report = agent.learn()
        self.assertEqual(len(report["loss"]), 8)
        self.assertEqual(learner._training_steps, 8)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(
            len({id(p) for g in learner._optimizer.param_groups for p in g["params"]}),
            sum(1 for p in learner._actor.parameters())
            + sum(1 for p in learner._critic.parameters()),
        )

    def test_reject_generic_replay_recurrent_history_and_wrong_metadata(self):
        learner = policy()
        with self.assertRaises(TypeError):
            learner.learn(BasicReplayBuffer(8))
        with self.assertRaises(TypeError):
            learner.set_history_summarization_module(
                LSTMHistorySummarizationModule(2, 1, 2, 4)
            )
        buffer = ContinuousPPORolloutBuffer(8, learner)
        with self.assertRaises(RuntimeError):
            buffer.push(
                torch.zeros(2),
                torch.zeros(1),
                1.0,
                False,
                False,
                next_state=torch.zeros(2),
            )
        learner.act(torch.zeros(2), space())
        with self.assertRaises(ValueError):
            buffer.push(
                torch.ones(2),
                torch.zeros(1),
                1.0,
                False,
                False,
                next_state=torch.zeros(2),
            )

    def test_populated_optimizer_checkpoint_continuation(self):
        torch.manual_seed(31)
        learner = policy()
        learner.learn_batch(batch())
        saved = io.BytesIO()
        torch.save(learner.state_dict(), saved)
        saved.seek(0)
        restored = policy()
        restored.load_state_dict(
            torch.load(saved, weights_only=True, map_location="cpu")
        )
        a = learner.learn_batch(batch())
        b = restored.learn_batch(batch())
        self.assertEqual(a, b)
        for p, q in zip(learner._actor.parameters(), restored._actor.parameters()):
            torch.testing.assert_close(p, q)
        self.assertEqual(restored._training_steps, 2)

    def test_real_pendulum_rollout_for_both_methods(self):
        torch.set_num_threads(1)
        for kind in [ContinuousProximalPolicyOptimization, RobustPolicyOptimization]:
            env = GymEnvironment("Pendulum-v1")
            learner = kind(
                3, env.action_space, hidden_dims=[8], batch_size=8, update_epochs=2
            )
            buffer = ContinuousPPORolloutBuffer(32, learner)
            agent = PearlAgent(learner, replay_buffer=buffer)
            observation, actions = env.reset(seed=24)
            agent.reset(observation, actions)
            before = [p.detach().clone() for p in learner._actor.parameters()]
            for _ in range(32):
                result = env.step(agent.act())
                self.assertTrue(np.isfinite(result.reward))
                agent.observe(result)
            result = agent.learn()
            self.assertTrue(all(math.isfinite(x) for x in result["loss"]))
            self.assertTrue(
                any(
                    not torch.equal(p, q)
                    for p, q in zip(learner._actor.parameters(), before)
                )
            )
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA")
    def test_cpu_cuda_populated_optimizer_movement(self):
        cpu = policy()
        cpu.learn_batch(batch())
        gpu = policy().cuda()
        gpu.load_state_dict(copy.deepcopy(cpu.state_dict()))
        a = cpu.learn_batch(batch())
        b = gpu.learn_batch(batch().to(torch.device("cuda")))
        self.assertAlmostEqual(a["loss"], b["loss"], places=5)
        for p, q in zip(cpu._actor.parameters(), gpu._actor.parameters()):
            torch.testing.assert_close(p, q.cpu(), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
