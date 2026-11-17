# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import unittest
from unittest.mock import patch

import torch
from pearl.replay_buffers.sequential_decision_making.prioritized_n_step_replay_buffer import (
    PrioritizedNStepReplayBuffer,
    PrioritizedNStepTransitionBatch,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestPrioritizedNStepReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = DiscreteActionSpace([torch.tensor([0]), torch.tensor([1])])

    def _push(
        self,
        replay: PrioritizedNStepReplayBuffer,
        state: int,
        reward: float,
        terminated: bool = False,
        truncated: bool = False,
        next_actions: DiscreteActionSpace | None = None,
    ) -> None:
        replay.push(
            state=torch.tensor([float(state)]),
            action=torch.tensor([state % 2]),
            reward=reward,
            next_state=torch.tensor([float(state + 1)]),
            terminated=terminated,
            truncated=truncated,
            curr_available_actions=self.actions,
            next_available_actions=(
                self.actions if next_actions is None else next_actions
            ),
            max_number_actions=2,
        )

    @staticmethod
    def _sample_slots(
        replay: PrioritizedNStepReplayBuffer, slots: list[int]
    ) -> PrioritizedNStepTransitionBatch:
        with patch(
            "torch.multinomial", return_value=torch.tensor(slots, dtype=torch.long)
        ):
            return replay.sample(len(slots))

    def test_three_step_return_and_terminal_prefixes(self) -> None:
        replay = PrioritizedNStepReplayBuffer(20, n_steps=3, discount_factor=0.5)
        self._push(replay, 0, 1.0)
        self._push(replay, 1, 2.0)
        self.assertEqual(len(replay), 0)
        self._push(replay, 2, 3.0)
        self.assertEqual(len(replay), 1)
        self._push(replay, 3, 4.0, terminated=True)
        self.assertEqual(len(replay), 4)
        batch = self._sample_slots(replay, [0, 1, 2, 3])
        torch.testing.assert_close(batch.reward, torch.tensor([2.75, 4.5, 5.0, 4.0]))
        torch.testing.assert_close(
            batch.bootstrap_discount, torch.tensor([0.125, 0.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            batch.state[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
        )
        torch.testing.assert_close(
            batch.next_state[:, 0], torch.tensor([3.0, 4.0, 4.0, 4.0])
        )
        self.assertEqual(batch.terminated.tolist(), [False, True, True, True])

    def test_truncation_bootstraps_and_keeps_final_action_mask(self) -> None:
        replay = PrioritizedNStepReplayBuffer(20, n_steps=3, discount_factor=0.5)
        final_actions = DiscreteActionSpace([torch.tensor([1])])
        self._push(replay, 0, 2.0)
        self._push(replay, 1, 4.0, truncated=True, next_actions=final_actions)
        batch = self._sample_slots(replay, [0, 1])
        torch.testing.assert_close(batch.reward, torch.tensor([4.0, 4.0]))
        torch.testing.assert_close(batch.bootstrap_discount, torch.tensor([0.25, 0.5]))
        torch.testing.assert_close(batch.next_state, torch.tensor([[2.0], [2.0]]))
        self.assertEqual(batch.terminated.tolist(), [False, False])
        self.assertEqual(batch.truncated.tolist(), [True, True])
        self.assertEqual(
            batch.next_available_actions[:, :, 0].tolist(), [[1.0, 0.0], [1.0, 0.0]]
        )
        self.assertEqual(
            batch.next_unavailable_actions_mask.tolist(), [[False, True], [False, True]]
        )
        self.assertEqual(
            batch.curr_unavailable_actions_mask.tolist(),
            [[False, False], [False, False]],
        )

    def test_windows_do_not_cross_episode_boundaries(self) -> None:
        replay = PrioritizedNStepReplayBuffer(20, n_steps=3, discount_factor=0.5)
        self._push(replay, 0, 1.0, terminated=True)
        self._push(replay, 100, 10.0)
        self._push(replay, 101, 20.0, truncated=True)
        batch = self._sample_slots(replay, [0, 1, 2])
        torch.testing.assert_close(batch.reward, torch.tensor([1.0, 20.0, 20.0]))
        torch.testing.assert_close(batch.state[:, 0], torch.tensor([0.0, 100.0, 101.0]))
        torch.testing.assert_close(
            batch.bootstrap_discount, torch.tensor([0.0, 0.25, 0.5])
        )

    def test_missing_terminal_observation_is_safe_in_mixed_batch(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10, n_steps=1)
        replay.push(
            state=torch.tensor([4.0]),
            action=torch.tensor([0]),
            reward=2.0,
            terminated=True,
            truncated=False,
            curr_available_actions=self.actions,
            next_state=None,
            next_available_actions=None,
        )
        self._push(replay, 10, 3.0)
        batch = self._sample_slots(replay, [0, 1])
        torch.testing.assert_close(batch.bootstrap_discount, torch.tensor([0.0, 0.99]))
        self.assertEqual(batch.next_state.shape, (2, 1))
        self.assertEqual(batch.next_available_actions.shape, (2, 2, 1))

    def test_missing_truncation_observation_rejected_without_insertion(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10)
        with self.assertRaisesRegex(ValueError, "Bootstrapping"):
            replay.push(
                state=torch.tensor([1.0]),
                action=torch.tensor([0]),
                reward=2.0,
                terminated=False,
                truncated=True,
                curr_available_actions=self.actions,
                next_available_actions=self.actions,
            )
        self.assertEqual(len(replay), 0)
        self.assertEqual(len(replay._pending), 0)

    def test_proportional_sampling_and_global_importance_normalization(self) -> None:
        replay = PrioritizedNStepReplayBuffer(
            10, n_steps=1, alpha=0.5, beta=1.0, priority_epsilon=1.0
        )
        self._push(replay, 0, 0.0)
        self._push(replay, 1, 0.0)
        replay.update_priorities(torch.tensor([0, 1]), torch.tensor([0.0, 3.0]))
        with patch("torch.multinomial", return_value=torch.tensor([1, 1, 0])) as sample:
            batch = replay.sample(3)
        probabilities = sample.call_args.args[0]
        torch.testing.assert_close(
            probabilities, torch.tensor([1 / 3, 2 / 3], dtype=torch.float64)
        )
        self.assertTrue(sample.call_args.kwargs["replacement"])
        torch.testing.assert_close(batch.weight, torch.tensor([0.5, 0.5, 1.0]))
        self.assertEqual(batch.indices.tolist(), [1, 1, 0])
        # Normalization must not depend on whether the least probable slot was sampled.
        high_priority_only = self._sample_slots(replay, [1])
        torch.testing.assert_close(high_priority_only.weight, torch.tensor([0.5]))

    def test_beta_zero_and_alpha_zero(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10, n_steps=1, alpha=0.0, beta=0.0)
        self._push(replay, 0, 0.0)
        self._push(replay, 1, 0.0)
        replay.update_priorities(torch.tensor([0, 1]), torch.tensor([0.0, 1000.0]))
        with patch("torch.multinomial", return_value=torch.tensor([0, 1])) as sample:
            batch = replay.sample(2)
        torch.testing.assert_close(
            sample.call_args.args[0], torch.tensor([0.5, 0.5], dtype=torch.float64)
        )
        torch.testing.assert_close(batch.weight, torch.ones(2))
        with self.assertRaises(ValueError):
            replay.beta = 1.1

    def test_duplicate_updates_use_maximum_and_are_order_independent(self) -> None:
        replay = PrioritizedNStepReplayBuffer(
            10, n_steps=1, alpha=1.0, priority_epsilon=1.0
        )
        self._push(replay, 0, 0.0)
        self._push(replay, 1, 0.0)
        replay.update_priorities(torch.tensor([0, 0, 1]), torch.tensor([5.0, 2.0, 2.0]))
        with patch("torch.multinomial", return_value=torch.tensor([0, 1])) as sample:
            replay.sample(2)
        expected = torch.tensor([2 / 3, 1 / 3], dtype=torch.float64)
        torch.testing.assert_close(sample.call_args.args[0], expected)
        replay.update_priorities(torch.tensor([1, 0, 0]), torch.tensor([2.0, 2.0, 5.0]))
        with patch("torch.multinomial", return_value=torch.tensor([0, 1])) as sample:
            replay.sample(2)
        torch.testing.assert_close(sample.call_args.args[0], expected)

    def test_priority_validation_is_atomic(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10, n_steps=1)
        self._push(replay, 0, 0.0)
        self._push(replay, 1, 0.0)
        invalid_updates = [
            (torch.tensor([0, 2]), torch.tensor([2.0, 3.0])),
            (torch.tensor([0.5]), torch.tensor([1.0])),
            (torch.tensor([0, 1]), torch.tensor([2.0, float("nan")])),
            (torch.tensor([0, 1]), torch.tensor([2.0, float("inf")])),
            (torch.tensor([0, 1]), torch.tensor([2.0, -1.0])),
        ]
        for indices, priorities in invalid_updates:
            with self.subTest(indices=indices, priorities=priorities):
                with self.assertRaises(ValueError):
                    replay.update_priorities(indices, priorities)
                torch.testing.assert_close(
                    replay._priorities[:2], torch.ones(2, dtype=torch.float64)
                )

    def test_capacity_and_sampling_with_replacement(self) -> None:
        replay = PrioritizedNStepReplayBuffer(2, n_steps=1)
        for state in range(3):
            self._push(replay, state, float(state))
        self.assertEqual(len(replay), 2)
        batch = self._sample_slots(replay, [0, 1])
        self.assertEqual(set(batch.state[:, 0].tolist()), {1.0, 2.0})
        self.assertEqual(len(replay.sample(20)), 20)

    def test_priority_epsilon_overflow_is_rejected(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10, n_steps=1, priority_epsilon=1e308)
        self._push(replay, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "remain finite"):
            replay.update_priorities(
                torch.tensor([0]), torch.tensor([1e308], dtype=torch.float64)
            )
        torch.testing.assert_close(
            replay._priorities[0], torch.tensor(1.0, dtype=torch.float64)
        )

    def test_clear_discards_pending_prefixes(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10, n_steps=3)
        self._push(replay, 0, 100.0)
        replay.clear()
        self._push(replay, 10, 1.0, terminated=True)
        batch = replay.sample(1)
        torch.testing.assert_close(batch.reward, torch.tensor([1.0]))
        torch.testing.assert_close(batch.state, torch.tensor([[10.0]]))
        replay.clear()
        self.assertEqual(len(replay), 0)
        with self.assertRaises(ValueError):
            replay.sample(1)

    def test_batch_to_preserves_replay_metadata(self) -> None:
        replay = PrioritizedNStepReplayBuffer(10, n_steps=1)
        self._push(replay, 0, 1.0)
        batch = replay.sample(2).to(torch.device("cpu"))
        self.assertIsInstance(batch, PrioritizedNStepTransitionBatch)
        self.assertEqual(batch.indices.dtype, torch.long)
        self.assertEqual(batch.indices.shape, (2,))
        torch.testing.assert_close(batch.bootstrap_discount, torch.full((2,), 0.99))

    def test_invalid_constructor_arguments(self) -> None:
        for kwargs in (
            {"size": 0},
            {"size": True},
            {"size": 2, "n_steps": 0},
            {"size": 2, "discount_factor": float("nan")},
            {"size": 2, "alpha": -0.5},
            {"size": 2, "beta": 2.0},
            {"size": 2, "priority_epsilon": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PrioritizedNStepReplayBuffer(**kwargs)


if __name__ == "__main__":
    unittest.main()
