#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

from pearl.core.common.neural_networks.value_networks import (
    DuelingStateActionValueNetwork,
    TwoTowerStateActionValueNetwork,
)
from pearl.core.common.pearl_agent import PearlAgent
from pearl.core.common.replay_buffer.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)

from pearl.core.contextual_bandits.environment.contextual_bandit_linear_synthetic_environment import (
    ContextualBanditLinearSyntheticEnvironment,
)
from pearl.core.contextual_bandits.policy_learners.disjoint_linear_bandit import (
    DisjointLinearBandit,
)
from pearl.core.contextual_bandits.policy_learners.exploration_module.linucb_exploration import (
    DisjointLinUCBExploration,
)
from pearl.core.contextual_bandits.replay_buffer.discrete_contextual_bandit_replay_buffer import (
    DiscreteContextualBanditReplayBuffer,
)

from pearl.core.sequential_decision_making.policy_learners.deep_q_learning import (
    DeepQLearning,
)

from pearl.gym.gym_environment import GymEnvironment
from pearl.online_learning.online_learning import online_learning_to_png_graph
from pearl.test.utils import create_random_batch
from pearl.utils.action_spaces import DiscreteActionSpace


class TestAgentWithPyTorch(unittest.TestCase):
    """
    A collection of Agent tests using PyTorch (this saves around 100 secs in test loading).
    For tests not involving PyTorch, use see test/without_pytorch.
    """

    def test_deep_td_learning_online_rl_sanity_check(self) -> None:
        # make sure E2E is fine
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=1,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )

    def test_deep_td_learning_online_rl_sanity_check_dueling(self) -> None:
        # make sure E2E is fine
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                network_type=DuelingStateActionValueNetwork,
                batch_size=1,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )

    def test_deep_td_learning_online_rl_two_tower_network(self) -> None:
        # make sure E2E is fine
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                network_type=TwoTowerStateActionValueNetwork,
                state_output_dim=64,
                action_output_dim=64,
                state_hidden_dims=[64],
                action_hidden_dims=[64],
                batch_size=1,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )

    def test_learn_batch(self) -> None:
        # simple test to ensure e2e run
        state_dim = 3
        action_dim = 3
        batch_size = 10
        batch, action_space = create_random_batch(action_dim, state_dim, batch_size)
        self.agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim,
                action_space,
                [64, 64],
            ),
        )
        self.agent.learn_batch(batch)

    def test_with_linear_contextual(self) -> None:
        """
        This is an integration test for ContextualBandit with ContextualBanditLinearSyntheticEnvironment
        """
        action_space = DiscreteActionSpace(range(3))
        feature_dim = 3

        agent = PearlAgent(
            policy_learner=DisjointLinearBandit(
                feature_dim=feature_dim,
                action_space=action_space,
                exploration_module=DisjointLinUCBExploration(alpha=0.1),
                batch_size=1,
            ),
            replay_buffer=DiscreteContextualBanditReplayBuffer(1),
        )
        env = ContextualBanditLinearSyntheticEnvironment(
            action_space=action_space,
            observation_dim=feature_dim,
        )

        regrets = []
        for _ in range(100):
            observation, action_space = env.reset()
            agent.reset(observation, action_space)
            action = agent.act()
            regret = env.get_regret(action)
            action_result = env.step(action)
            agent.observe(action_result)
            agent.learn()
            regrets.append(regret.squeeze().item())

        # to test learning ability of linear contextual bandits we check
        # that the regret is decreasing over learning steps
        self.assertTrue(sum(regrets[10:]) >= sum(regrets[-10:]))
