#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import copy
import unittest

import torch
from pearl.policy_learners.contextual_bandits.disjoint_bandit import (
    DisjointBanditContainer,
)
from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (  # noqa E501
    ThompsonSamplingExplorationLinearDisjoint,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (  # noqa E501
    DisjointUCBExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestDisjointLinearBandits(unittest.TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        action_space = DiscreteActionSpace(list(range(3)))
        policy_learner = DisjointLinearBandit(
            feature_dim=2,
            action_space=action_space,
            # UCB score == rewards
            exploration_module=DisjointUCBExploration(alpha=0),
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
        for _ in range(1000):
            policy_learner.learn_batch(batch)

        self.policy_learner = policy_learner
        self.batch = batch
        self.action_space = action_space

    def test_learn_batch(self) -> None:
        for i, action in enumerate(self.batch.action):
            action = action.item()
            # check if linear regression works
            self.assertTrue(
                torch.allclose(
                    self.policy_learner._linear_regressions[action](
                        self.batch.state[i]
                    ),
                    self.batch.reward[i],
                    atol=1e-1,
                )
            )

    def test_ucb_act(self) -> None:
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module
        action_space = self.action_space
        batch = self.batch

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
        # set a different alpha value to increase weight of sigma
        policy_learner.exploration_module = DisjointUCBExploration(alpha=10000)
        # observe state [1,1] for action 1 and 2 many times,
        # this will increase sigma of action0
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
        for _ in range(1000):
            policy_learner.learn_batch(batch)
        self.assertEqual(
            0,
            policy_learner.act(
                subjective_state=torch.tensor([1.0, 1.0]), action_space=action_space
            ),
        )

    def test_thompson_sampling_disjoint_act(self) -> None:
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module
        policy_learner.exploration_module = ThompsonSamplingExplorationLinearDisjoint()
        action_space = self.action_space
        batch = self.batch

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
            all(a in range(0, action_space.n) for a in selected_actions.tolist())
        )

    def test_ucb_action_vector(self) -> None:
        """
        This is to test discreate action space, but each action has a action vector
        """
        state_dim = 5
        action_dim = 3
        action_count = 3
        batch_size = 10

        # pyre-fixme[6]: For 1st argument expected `List[typing.Any]` but got `Tensor`.
        action_space = DiscreteActionSpace(torch.randn(action_count, action_dim))
        policy_learner = DisjointLinearBandit(
            feature_dim=state_dim + action_dim,
            action_space=action_space,
            exploration_module=DisjointUCBExploration(alpha=0.1),
        )
        batch = TransitionBatch(
            state=torch.randn(batch_size, state_dim),
            action=torch.randint(
                low=1, high=3, size=(batch_size,)
            ),  # this is action index
            reward=torch.randn(batch_size),
            weight=torch.ones(batch_size),
        )
        action = policy_learner.act(
            subjective_state=batch.state[0], action_space=action_space
        )
        self.assertEqual(action.shape, ())
        action = policy_learner.act(
            subjective_state=batch.state, action_space=action_space
        )
        self.assertEqual(action.shape, torch.Size([batch_size]))


class TestDisjointBanditContainerLinearBandits(unittest.TestCase):
    def setUp(self) -> None:
        num_arms = 3
        action_space = DiscreteActionSpace(list(range(num_arms)))
        feature_dim = 2
        policy_learner = DisjointBanditContainer(
            feature_dim=feature_dim,
            arm_bandits=[
                LinearBandit(feature_dim=feature_dim) for _ in range(num_arms)
            ],
            exploration_module=DisjointUCBExploration(alpha=0),
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
        for _ in range(1000):
            policy_learner.learn_batch(batch)

        self.policy_learner = policy_learner
        self.batch = batch
        self.action_space = action_space

    def test_learn_batch(self) -> None:
        for i, action in enumerate(self.batch.action):
            action = action.item()
            # check if each arm model works
            self.assertTrue(
                torch.allclose(
                    self.policy_learner._arm_bandits[action].model(self.batch.state[i]),
                    self.batch.reward[i],
                    atol=1e-1,
                )
            )

    def test_ucb_act(self) -> None:
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module
        action_space = self.action_space
        batch = self.batch

        # since alpha = 0, act should return action with highest reward
        # single state
        self.assertEqual(
            2,
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]),
                available_action_space=action_space,
            ),
        )
        # batch state
        self.assertTrue(
            torch.all(
                policy_learner.act(
                    subjective_state=batch.state, available_action_space=action_space
                )
                == 2
            )
        )
        # set a different alpha value to increase weight of sigma
        policy_learner.exploration_module = DisjointUCBExploration(alpha=10000)
        # observe state [1,1] for action 1 and 2 many times,
        # this will increase sigma of action0
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
        for _ in range(1000):
            policy_learner.learn_batch(batch)
        self.assertEqual(
            0,
            policy_learner.act(
                subjective_state=torch.tensor([1.0, 1.0]),
                available_action_space=action_space,
            ),
        )

    def test_thompson_sampling_disjoint_act(self) -> None:
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module
        policy_learner.exploration_module = ThompsonSamplingExplorationLinearDisjoint()
        action_space = self.action_space
        batch = self.batch

        # test act
        self.assertTrue(
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]),
                available_action_space=action_space,
            )
            in range(0, action_space.n)
        )

        selected_actions = policy_learner.act(
            subjective_state=batch.state, available_action_space=action_space
        )
        self.assertTrue(selected_actions.shape[0] == batch.state.shape[0])

        self.assertTrue(
            all(a in range(0, action_space.n) for a in selected_actions.tolist())
        )

    def test_ucb_action_vector(self) -> None:
        """
        This is to test discreate action space, but each action has a action vector
        """
        state_dim = 5
        action_count = 3
        batch_size = 10

        action_space = DiscreteActionSpace(list(range(action_count)))
        policy_learner = DisjointBanditContainer(
            feature_dim=state_dim,
            arm_bandits=[
                LinearBandit(feature_dim=state_dim) for _ in range(action_count)
            ],
            exploration_module=DisjointUCBExploration(alpha=0.1),
        )
        batch = TransitionBatch(
            state=torch.randn(batch_size, state_dim),
            action=torch.randint(
                low=0, high=(action_count - 1), size=(batch_size,)
            ),  # this is action index
            reward=torch.randn(batch_size),
            weight=torch.ones(batch_size),
        )
        action = policy_learner.act(
            subjective_state=batch.state[0], available_action_space=action_space
        )
        self.assertEqual(action.shape, ())
        action = policy_learner.act(
            subjective_state=batch.state, available_action_space=action_space
        )
        self.assertEqual(action.shape, torch.Size([batch_size]))

    def test_get_scores_linear(self) -> None:
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module
        alpha = 3.0
        policy_learner.exploration_module = DisjointUCBExploration(alpha=alpha)
        # action_space = self.action_space
        # batch = self.batch
        batch_size = len(self.batch)

        # get scores
        scores = policy_learner.get_scores(
            subjective_state=self.batch.state, action_space=self.action_space
        )
        self.assertEqual(scores.shape, torch.Size([batch_size, self.action_space.n]))

        # test that scores have the correct values
        features = self.batch.state
        expected_scores = []
        for i in range(self.action_space.n):
            model = policy_learner.models[i]  # model for arm i
            mus = model(features)
            sigmas = model.calculate_sigma(features)
            expected_scores.append(mus + alpha * sigmas)
        expected_scores = torch.stack(expected_scores, dim=1)
        self.assertTrue(torch.allclose(scores, expected_scores, atol=1e-1))
