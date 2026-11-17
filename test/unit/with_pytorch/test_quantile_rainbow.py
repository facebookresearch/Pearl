# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import unittest
from unittest.mock import patch

import torch
from pearl.api.action_result import ActionResult
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.quantile_rainbow import (
    QuantileRainbow,
)
from pearl.replay_buffers.sequential_decision_making.prioritized_n_step_replay_buffer import (
    PrioritizedNStepReplayBuffer,
    PrioritizedNStepTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.risk_sensitive_safety_modules import (
    QuantileNetworkMeanVarianceSafetyModule,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class _TableNetwork(torch.nn.Module):
    def __init__(self, table: list[list[float]]) -> None:
        super().__init__()
        self.register_buffer("table", torch.tensor(table, dtype=torch.float32))
        self.register_buffer("quantiles", torch.linspace(0, 1, len(table[0]) + 1))

    def get_q_value_distribution(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.table[actions.long().squeeze(-1)]


def _space() -> DiscreteActionSpace:
    return DiscreteActionSpace([torch.tensor([0]), torch.tensor([1])])


def _learner(**options: object) -> QuantileRainbow:
    settings = dict(
        state_dim=2,
        action_space=_space(),
        hidden_dims=[],
        num_quantiles=2,
        noisy=False,
        dueling=False,
        learning_rate=0.0,
        max_grad_norm=None,
    )
    settings.update(options)
    return QuantileRainbow(**settings)


def _batch(terminated: bool = False) -> TransitionBatch:
    return TransitionBatch(
        state=torch.zeros(1, 2),
        action=torch.zeros(1, 1),
        reward=torch.ones(1),
        terminated=torch.tensor([terminated]),
        truncated=torch.tensor([False]),
        next_state=torch.ones(1, 2),
        next_available_actions=torch.tensor([[[0], [1]]]),
        next_unavailable_actions_mask=torch.tensor([[False, False]]),
    )


class TestQuantileRainbow(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)

    def test_double_selection_uses_online_and_target_evaluation(self) -> None:
        learner = _learner()
        learner._Q = _TableNetwork([[10, 12], [1, 3]])
        learner._Q_target = _TableNetwork([[2, 4], [8, 10]])
        batch = _batch()
        # Reordered candidate slots are not themselves action IDs.
        batch.next_available_actions = torch.tensor([[[1], [0]]])
        actual = learner._get_next_state_quantiles(batch, 1)
        torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0]]))
        learner._double_q = False
        torch.testing.assert_close(
            learner._get_next_state_quantiles(batch, 1), torch.tensor([[8.0, 10.0]])
        )

    def test_masked_best_action_and_padding_are_excluded(self) -> None:
        learner = _learner()
        learner._Q = _TableNetwork([[10, 12], [1, 3]])
        learner._Q_target = _TableNetwork([[2, 4], [8, 10]])
        batch = _batch()
        batch.next_available_actions = torch.tensor([[[99], [1]]])
        batch.next_unavailable_actions_mask = torch.tensor([[True, False]])
        torch.testing.assert_close(
            learner._get_next_state_quantiles(batch, 1), torch.tensor([[8.0, 10.0]])
        )

    def test_risk_metric_is_used_for_online_selection(self) -> None:
        learner = _learner()
        learner._Q = _TableNetwork([[4, 4], [0, 12]])
        learner._Q_target = _TableNetwork([[1, 2], [20, 21]])
        learner.safety_module = QuantileNetworkMeanVarianceSafetyModule(0.2)
        torch.testing.assert_close(
            learner._get_next_state_quantiles(_batch(), 1), torch.tensor([[1.0, 2.0]])
        )

    def test_terminal_rows_do_not_require_next_states(self) -> None:
        learner = _learner()
        batch = _batch(terminated=True)
        batch.next_state = None
        batch.next_available_actions = None
        torch.testing.assert_close(
            learner._get_next_state_quantiles(batch, 1), torch.zeros(1, 2)
        )
        self.assertTrue(
            torch.isfinite(torch.tensor(learner.learn_batch(batch)["loss"]))
        )

    def test_mixed_terminal_rows_skip_nonfinite_placeholders(self) -> None:
        learner = _learner()
        learner._Q = _TableNetwork([[10, 12], [1, 3]])
        learner._Q_target = _TableNetwork([[2, 4], [8, 10]])
        batch = TransitionBatch(
            state=torch.zeros(2, 2),
            action=torch.zeros(2, 1),
            reward=torch.ones(2),
            terminated=torch.tensor([True, False]),
            next_state=torch.tensor([[float("nan"), float("nan")], [0.0, 0.0]]),
            next_available_actions=torch.tensor([[[99], [99]], [[0], [1]]]),
            next_unavailable_actions_mask=torch.tensor([[True, True], [False, False]]),
        )
        torch.testing.assert_close(
            learner._get_next_state_quantiles(batch, 2),
            torch.tensor([[0.0, 0.0], [2.0, 4.0]]),
        )

    def test_nonterminal_all_masked_actions_are_rejected(self) -> None:
        batch = _batch()
        batch.next_unavailable_actions_mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "at least one available action"):
            _learner()._get_next_state_quantiles(batch, 1)

    def test_importance_weights_do_not_change_priorities(self) -> None:
        learner = _learner()
        with torch.no_grad():
            for parameter in learner._Q.parameters():
                parameter.zero_()
        batch = TransitionBatch(
            state=torch.zeros(2, 2),
            action=torch.zeros(2, 1),
            reward=torch.tensor([1.0, 3.0]),
            terminated=torch.ones(2, dtype=torch.bool),
            weight=torch.tensor([0.25, 1.0]),
        )
        report, priorities = learner._learn_batch_with_priorities(batch)
        # For two zero quantiles with midpoints .25/.75, losses are .5 and 2.5.
        torch.testing.assert_close(priorities, torch.tensor([0.5, 2.5]))
        self.assertAlmostEqual(report["loss"], 1.3125)
        self.assertFalse(priorities.requires_grad)
        batch.weight = torch.ones(2)
        report, priorities_again = learner._learn_batch_with_priorities(batch)
        torch.testing.assert_close(priorities_again, priorities)
        self.assertAlmostEqual(report["loss"], 1.5)

    def test_shortened_truncation_bootstraps_with_actual_discount(self) -> None:
        learner = _learner(num_quantiles=1, discount_factor=0.5)
        with torch.no_grad():
            for parameter in learner._Q.parameters():
                parameter.zero_()
            for parameter in learner._Q_target.parameters():
                parameter.zero_()
            learner._Q_target.quantile_head.bias.fill_(4)
        batch = PrioritizedNStepTransitionBatch(
            state=torch.zeros(1, 2),
            action=torch.zeros(1, 1),
            reward=torch.tensor([2.0]),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([True]),
            next_state=torch.ones(1, 2),
            next_available_actions=torch.tensor([[[0], [1]]]),
            indices=torch.tensor([0]),
            bootstrap_discount=torch.tensor([0.25]),
            weight=torch.ones(1),
        )
        # Target is 2 + .25 * 4 = 3; tau=.5, Huber(3)=2.5 -> loss 1.25.
        self.assertAlmostEqual(learner.learn_batch(batch)["loss"], 1.25)

    def test_target_update_occurs_after_exact_number_of_optimizer_updates(self) -> None:
        learner = _learner(learning_rate=0.01, target_update_freq=2)
        before = [p.clone() for p in learner._Q_target.parameters()]
        learner.learn_batch(_batch(terminated=True))
        self.assertEqual(learner._training_steps, 1)
        for initial, current in zip(before, learner._Q_target.parameters()):
            torch.testing.assert_close(initial, current, rtol=0, atol=0)
        learner.learn_batch(_batch(terminated=True))
        self.assertEqual(learner._training_steps, 2)
        for online, target in zip(
            learner._Q.parameters(), learner._Q_target.parameters()
        ):
            torch.testing.assert_close(online, target, rtol=0, atol=0)
            self.assertIsNone(target.grad)

    def test_exploit_suppresses_noise_and_restores_previous_setting(self) -> None:
        learner = _learner(noisy=True, dueling=True)
        previous = learner._Q.noise_enabled
        observed = []
        handle = learner._Q.register_forward_pre_hook(
            lambda module, args: observed.append(module.noise_enabled)
        )
        # get_q_value_distribution invokes forward directly, so record at the head.
        handle.remove()
        head = learner._Q.value_head
        handle = head.register_forward_pre_hook(
            lambda module, args: observed.append(learner._Q.noise_enabled)
        )
        actions = [learner.act(torch.ones(2), _space(), exploit=True) for _ in range(8)]
        handle.remove()
        self.assertEqual(observed, [False] * 8)
        self.assertTrue(all(torch.equal(action, actions[0]) for action in actions))
        self.assertEqual(learner._Q.noise_enabled, previous)

    def test_checkpoint_restores_optimizer_and_update_counter(self) -> None:
        learner = _learner(learning_rate=0.01, target_update_freq=2)
        learner.learn_batch(_batch(terminated=True))
        checkpoint = io.BytesIO()
        torch.save(learner.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = _learner(learning_rate=0.01, target_update_freq=2)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.assertEqual(restored._training_steps, 1)
        self.assertEqual(restored.compare(learner), "")
        learner.learn_batch(_batch(terminated=True))
        restored.learn_batch(_batch(terminated=True))
        for actual, expected in zip(restored._Q.parameters(), learner._Q.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(
            restored._Q_target.parameters(), learner._Q_target.parameters()
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_rejects_algorithm_configuration_changes(self) -> None:
        learner = _learner()
        restored = _learner(double_q=False)
        with self.assertRaisesRegex(ValueError, "configuration"):
            restored.load_state_dict(copy.deepcopy(learner.state_dict()))

    def test_trained_pearl_agent_checkpoint_restores_history_optimizer_group(
        self,
    ) -> None:
        agent = PearlAgent(
            policy_learner=_learner(learning_rate=0.01),
            replay_buffer=PrioritizedNStepReplayBuffer(8),
        )
        agent.learn_batch(_batch(terminated=True).to(agent.device))
        restored = PearlAgent(
            policy_learner=_learner(learning_rate=0.01),
            replay_buffer=PrioritizedNStepReplayBuffer(8),
        )
        restored.load_state_dict(copy.deepcopy(agent.state_dict()))
        self.assertEqual(restored.policy_learner._training_steps, 1)
        self.assertEqual(len(restored.policy_learner._optimizer.param_groups), 2)
        agent.learn_batch(_batch(terminated=True).to(agent.device))
        restored.learn_batch(_batch(terminated=True).to(restored.device))
        for expected, actual in zip(
            agent.policy_learner._Q.parameters(),
            restored.policy_learner._Q.parameters(),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pearl_agent_act_observe_learn_and_priority_feedback(self) -> None:
        learner = _learner(noisy=True, dueling=True, learning_rate=0.001)
        replay = PrioritizedNStepReplayBuffer(32, n_steps=3, discount_factor=0.99)
        agent = PearlAgent(policy_learner=learner, replay_buffer=replay)
        space = _space()
        agent.reset(torch.zeros(2), space)
        with patch.object(
            replay, "update_priorities", wraps=replay.update_priorities
        ) as update:
            for step in range(5):
                agent.act()
                agent.observe(
                    ActionResult(
                        observation=torch.tensor([float(step + 1), 0.0]),
                        reward=1.0,
                        terminated=False,
                        truncated=step == 4,
                        available_action_space=space,
                    )
                )
                report = agent.learn()
                if report:
                    self.assertTrue(
                        all(math_value >= 0 for math_value in report["loss"])
                    )
            self.assertEqual(update.call_count, 3)
        self.assertEqual(len(replay), 5)
        self.assertEqual(learner._training_steps, 3)

    def test_replay_discount_mismatch_is_rejected(self) -> None:
        replay = PrioritizedNStepReplayBuffer(4, n_steps=1, discount_factor=0.5)
        replay.push(
            state=torch.zeros(2),
            action=torch.tensor([0]),
            reward=1.0,
            terminated=True,
            truncated=False,
            curr_available_actions=_space(),
            max_number_actions=2,
        )
        with self.assertRaisesRegex(ValueError, "same discount_factor"):
            _learner().learn(replay)

    def test_non_catalog_actions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordered scalar catalog"):
            _learner(
                action_space=DiscreteActionSpace(
                    [torch.tensor([-1]), torch.tensor([1])]
                )
            )
        with self.assertRaisesRegex(ValueError, "unique IDs"):
            _learner().act(
                torch.zeros(2),
                DiscreteActionSpace([torch.tensor([0]), torch.tensor([0])]),
            )


if __name__ == "__main__":
    unittest.main()
