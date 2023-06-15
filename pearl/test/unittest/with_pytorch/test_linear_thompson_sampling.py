#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.contextual_bandits.disjoint_linear_bandit import DisjointLinearBandit
from pearl.policy_learners.exploration_module.thompson_sampling_exploration import (
    ThompsonSamplingExplorationLinearDisjoint,
)
from pearl.replay_buffer.transition import TransitionBatch
from pearl.utils.action_spaces import DiscreteActionSpace


class TestLinThompsonSampling(unittest.TestCase):
    def test_disjoint_lin_thompson_sampling_learn_batch(self) -> None:
        action_space = DiscreteActionSpace(range(3))
        policy_learner = DisjointLinearBandit(
            feature_dim=2,
            action_count=action_space.n,
            exploration_module=ThompsonSamplingExplorationLinearDisjoint(),
        )
        # y0 = x1  + x2
        # y1 = 2x1 + x2
        # y2 = 2x1 + 2x2
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, 3.0],
                    [2.0, 3.0],
                    [1.0, 5.0],
                    [0.5, 3.0],
                    [1.8, 5.1],
                ]
            ),
            action=torch.tensor(
                [0, 0, 1, 1, 2, 2],
            ),
            reward=torch.tensor([3.0, 4.0, 7.0, 7.0, 7.0, 13.8]),
            weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )

        # test learn
        policy_learner.learn_batch(batch)
        for i, action in enumerate(batch.action):
            action = action.item()
            # check if linear regression works
            self.assertTrue(
                torch.allclose(
                    policy_learner._linear_regressions[action](batch.state[i]),
                    batch.reward[i],
                    atol=1e-1,
                )
            )

        # test act
        self.assertTrue(
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]), action_space=action_space
            )
            in range(0, action_space.n)
        )

        selected_actions = policy_learner.act(
            subjective_state=batch.state, action_space=action_space
        )
        self.assertTrue(selected_actions.shape[0] == batch.state.shape[0])

        self.assertTrue(
            all([a in range(0, action_space.n) for a in selected_actions.tolist()])
        )
