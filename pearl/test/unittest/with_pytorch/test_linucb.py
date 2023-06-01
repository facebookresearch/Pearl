#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import torch
from pearl.contextual_bandits.disjoint_linUCB import DisjointLinUCB
from pearl.contextual_bandits.linUCB import LinUCB
from pearl.policy_learners.exploration_module.no_exploration import NoExploration
from pearl.replay_buffer.transition import TransitionBatch
from pearl.utils.action_spaces import DiscreteActionSpace

logger = logging.getLogger(__name__)


class TestLinUCB(unittest.TestCase):
    def test_disjoint_lin_ucb_learn_batch(self) -> None:
        policy_learner = DisjointLinUCB(
            state_dim=2,
            action_space=DiscreteActionSpace(range(3)),
            exploration_module=NoExploration(),
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
            self.assertTrue(
                torch.allclose(
                    policy_learner._linear_regressions[action](batch.state[i]),
                    batch.reward[i],
                    atol=1e-4,
                )
            )

    def test_lin_ucb_learn_batch(self) -> None:
        policy_learner = LinUCB(
            feature_dim=4,
            exploration_module=NoExploration(),
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

    def test_lin_ucb_uncertainty(self) -> None:
        """
        Since A is init with zeros, the uncertainty should be proportional to number of obs
        """

        policy_learner = LinUCB(
            feature_dim=2,
            exploration_module=NoExploration(),
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
        _ = policy_learner._linear_regression(
            torch.cat([batch.state[0], batch.action[0]])
        )  # predict reward

        # test uncertainty of policy_learner (LinUCB)
        A = policy_learner._linear_regression._A
        A_inv = torch.linalg.inv(A)
        x = torch.cat([batch.state, batch.action], dim=1)
        uncertainty = torch.diagonal(x @ A_inv @ x.t())
        logger.info(uncertainty)

        # the 2nd arm's uncertainty is 10 times 1st arm's uncertainty
        uncertainty_ratio = torch.tensor(uncertainty[-1] / uncertainty[0])
        self.assertTrue(
            torch.allclose(
                uncertainty_ratio,
                torch.tensor(10.0),  # the 1st arm occured 10 times than 2nd arm
                rtol=0.01,
            )
        )

        return
