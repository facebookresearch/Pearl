#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.contextual_bandits.disjoint_linear_bandit import DisjointLinearBandit
from pearl.contextual_bandits.disjoint_linucb_exploration import (
    DisjointLinUCBExploration,
)
from pearl.contextual_bandits.linear_bandit import LinearBandit
from pearl.contextual_bandits.linear_regression import (
    AvgWeightLinearRegression,
    LinearRegression,
)
from pearl.replay_buffer.transition import TransitionBatch
from pearl.utils.action_spaces import DiscreteActionSpace


class TestLinUCB(unittest.TestCase):
    def test_disjoint_lin_ucb_learn_batch(self) -> None:
        action_space = DiscreteActionSpace(range(3))
        policy_learner = DisjointLinearBandit(
            feature_dim=2,
            action_count=action_space.n,
            # UCB score == rewards
            exploration_module=DisjointLinUCBExploration(alpha=0),
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
        policy_learner.learn_batch(batch)
        for i, action in enumerate(batch.action):
            action = action.item()
            # check if linear regression works
            self.assertTrue(
                torch.allclose(
                    policy_learner._linear_regressions[action](batch.state[i]),
                    batch.reward[i],
                    atol=1e-4,
                )
            )
        # since alpha = 0, act should return action with highest reward
        # single state
        self.assertEqual(
            2,
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]), action_space=action_space
            ),
        )
        # batch state
        self.assertTrue(
            torch.all(
                policy_learner.act(
                    subjective_state=batch.state, action_space=action_space
                )
                == 2
            )
        )
        # set a different alpha value to increase uncertainty value
        policy_learner.exploration_module = DisjointLinUCBExploration(alpha=10)
        # observe state [1,1] for action 1 and 2 many times, this will increase uncertainty of action0
        # on this state, and give us act(1,1) -> 0
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            ),
            action=torch.tensor(
                [1, 2],
            ),
            reward=torch.tensor([2.0, 3.0]),
            weight=torch.tensor([1.0, 1.0]),
        )
        for _ in range(10):
            policy_learner.learn_batch(batch)
        self.assertEqual(
            0,
            policy_learner.act(
                subjective_state=torch.tensor([1.0, 1.0]), action_space=action_space
            ),
        )

    def test_lin_ucb_learn_batch(self) -> None:
        policy_learner = LinearBandit(
            feature_dim=4,
            exploration_module=DisjointLinUCBExploration(alpha=0),
        )
        # y = sum of state + sum of action
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, 3.0],
                    [2.0, 2.0],
                    [2.0, 3.0],
                ]
            ),
            action=torch.tensor(
                [
                    [2.0, 2.0],
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 3.0],
                ]
            ),
            reward=torch.tensor([7.0, 7.0, 9.0, 9.0]),
            weight=torch.tensor([1, 1, 1, 1]),
        )
        policy_learner.learn_batch(batch)
        # a single input
        self.assertTrue(
            torch.allclose(
                policy_learner._linear_regression(
                    torch.cat([batch.state[0], batch.action[0]])
                ),
                batch.reward[0],
                atol=1e-4,
            )
        )
        # a batch input
        self.assertTrue(
            torch.allclose(
                policy_learner._linear_regression(
                    torch.cat([batch.state, batch.action], dim=1)
                ),
                batch.reward,
                atol=1e-4,
            )
        )
        # with ucb_alpha == 0, ucb scores == rewards
        # we view action space as 1, in order to get ucb scores for given feature
        ucb_scores = policy_learner.get_scores(
            torch.cat([batch.state, batch.action], dim=1)
        )
        self.assertTrue(
            torch.allclose(
                ucb_scores,
                batch.reward,
                atol=1e-4,
            )
        )
        self.assertEqual(ucb_scores.shape, batch.reward.shape)

    def test_lin_ucb_uncertainty(self) -> None:
        """
        Since A is init with zeros, the uncertainty should be proportional to number of obs
        """

        policy_learner = LinearBandit(
            feature_dim=2,
            exploration_module=DisjointLinUCBExploration(alpha=1),
        )

        # 1st arm = [1, 0]
        # 2nd arm = [0, 1]
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],  # repeat 1st arm ten times
                    [0.0],
                ]
            ),
            action=torch.tensor(
                [
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [1.0],
                ]
            ),
            reward=torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0]
            ),
            weight=torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ),
        )

        policy_learner.learn_batch(batch)

        # test uncertainty of policy_learner (LinUCB)
        features = torch.cat([batch.state, batch.action], dim=1)
        # input is (batch_size, 2)
        # expect output is (batch_size,)
        ucb_scores = policy_learner.get_scores(features)
        self.assertEqual(ucb_scores.shape, batch.reward.shape)
        A = policy_learner._linear_regression._A
        A_inv = torch.linalg.inv(A)
        uncertainty = torch.diagonal(features @ A_inv @ features.t())

        # the 2nd arm's uncertainty is 10 times 1st arm's uncertainty
        uncertainty_ratio = torch.tensor(uncertainty[-1] / uncertainty[0])
        self.assertTrue(
            torch.allclose(
                uncertainty_ratio,
                torch.tensor(10.0),  # the 1st arm occured 10 times than 2nd arm
                rtol=0.01,
            )
        )

    def test_linear_regression_random(self) -> None:
        feature_dim = 15
        batch_size = feature_dim * 4  # it is important to have enough data for training

        def single_test(linear_regression_class):
            linear_regression = linear_regression_class(feature_dim=feature_dim)
            losses = []
            feature = torch.randn(batch_size, feature_dim)
            reward = feature.sum(-1)
            weight = torch.ones(batch_size)
            for _ in range(10):
                loss = (linear_regression(feature) - reward) ** 2
                losses.append(loss.mean().item())
                linear_regression.train(x=feature, y=reward, weight=weight)
            losses.append(loss.mean().item())

            self.assertGreater(sum(losses[:5]), sum(losses[-5:]))
            self.assertGreater(1e-2, losses[-1])

        single_test(AvgWeightLinearRegression)
        single_test(LinearRegression)
