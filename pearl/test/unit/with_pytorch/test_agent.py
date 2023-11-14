#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

from pearl.neural_networks.common.value_networks import (
    DuelingQValueNetwork,
    TwoTowerQValueNetwork,
)
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)

from pearl.policy_learners.contextual_bandits.square_cb import SquareCB

from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
)

from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers.contextual_bandits.discrete_contextual_bandit_replay_buffer import (
    DiscreteContextualBanditReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import (
    online_learning_to_png_graph,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace

from pearl.utils.instantiations.environments.contextual_bandit_linear_synthetic_environment import (
    ContextualBanditLinearSyntheticEnvironment,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment


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

    def test_conservative_deep_td_learning_online_rl_sanity_check(self) -> None:
        # make sure E2E is fine for cql loss
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                is_conservative=True,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )

    def test_deep_td_learning_online_rl_sanity_check_dueling(
        self,
        number_of_episodes: int = 10,
        batch_size: int = 128,
    ) -> None:
        # make sure E2E is fine
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                network_type=DuelingQValueNetwork,
                batch_size=batch_size,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=number_of_episodes, learn_after_episode=True
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
                network_type=TwoTowerQValueNetwork,
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

    def test_with_linear_contextual(self) -> None:
        """
        This is an integration test for ContextualBandit with ContextualBanditLinearSyntheticEnvironment
        """
        # pyre-fixme[6]: For 1st argument expected `List[typing.Any]` but got `range`.
        action_space = DiscreteActionSpace(range(3))
        feature_dim = 3

        agent = PearlAgent(
            policy_learner=DisjointLinearBandit(
                feature_dim=feature_dim,
                action_space=action_space,
                exploration_module=DisjointUCBExploration(alpha=0.1),
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
            # pyre-fixme[16]: `Number` has no attribute `squeeze`.
            regrets.append(regret.squeeze().item())

        # to test learning ability of linear contextual bandits we check
        # that the regret is decreasing over learning steps
        self.assertTrue(sum(regrets[10:]) >= sum(regrets[-10:]))

    def test_squarecb(self) -> None:
        """
        This is an integration test for SquareCB
        The parameter gamma should be set proportionally to sqrt{A T / d}
        (see https://arxiv.org/pdf/2002.04926.pdf and discussion after Theorem 1)
        """
        # pyre-fixme[6]: For 1st argument expected `List[typing.Any]` but got `range`.
        action_space = DiscreteActionSpace(range(3))
        feature_dim = 3

        agent = PearlAgent(
            policy_learner=SquareCB(
                feature_dim=feature_dim,
                action_space=action_space,
                gamma=10,
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
            # pyre-fixme[16]: `Number` has no attribute `squeeze`.
            regrets.append(regret.squeeze().item())

        # to test learning ability of linear contextual bandits we check
        # that the regret is decreasing over learning steps
        self.assertTrue(sum(regrets[10:]) >= sum(regrets[-10:]))
