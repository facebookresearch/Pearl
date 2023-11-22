#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)

from pearl.neural_networks.common.value_networks import (
    DuelingQValueNetwork,
    TwoTowerQValueNetwork,
)
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)

from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
)

from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.tabular_q_learning import (
    TabularQLearning,
)
from pearl.replay_buffers.contextual_bandits.discrete_contextual_bandit_replay_buffer import (
    DiscreteContextualBanditReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import (
    episode_return,
    online_learning,
    online_learning_to_png_graph,
)
from pearl.utils.instantiations.action_spaces.discrete import DiscreteActionSpace

from pearl.utils.instantiations.environments.contextual_bandit_linear_synthetic_environment import (
    ContextualBanditLinearSyntheticEnvironment,
)
from pearl.utils.instantiations.environments.environments import (
    FixedNumberOfStepsEnvironment,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.environments.reward_is_equal_to_ten_times_action_contextual_bandit_environment import (
    RewardIsEqualToTenTimesActionContextualBanditEnvironment,
)


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
                state_dim=env.observation_space.shape[0],  # pyre-ignore[16] (assumes Box)
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=1,
            ),
            action_representation_module=OneHotActionTensorRepresentationModule(
                max_actions=env.action_space.n
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
                state_dim=env.observation_space.shape[0],  # pyre-ignore[16] (assumes Box)
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                is_conservative=True,
            ),
            action_representation_module=OneHotActionTensorRepresentationModule(
                max_actions=env.action_space.n
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
                env.observation_space.shape[0],  # pyre-ignore[16] (assumes Box)
                env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                network_type=DuelingQValueNetwork,
                batch_size=batch_size,
            ),
            action_representation_module=OneHotActionTensorRepresentationModule(
                max_actions=env.action_space.n
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
                state_dim=env.observation_space.shape[0],  # pyre-ignore[16] (assumes Box)
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
            action_representation_module=OneHotActionTensorRepresentationModule(
                max_actions=env.action_space.n
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )

    def test_with_linear_contextual(self) -> None:
        """
        This is an integration test for ContextualBandit with
        ContextualBanditLinearSyntheticEnvironment.
        """
        action_space = DiscreteActionSpace(
            actions=[torch.tensor([a]) for a in range(3)]
        )
        observation_dim = 3

        agent = PearlAgent(
            policy_learner=DisjointLinearBandit(
                feature_dim=observation_dim + 1,
                action_space=action_space,
                exploration_module=DisjointUCBExploration(alpha=0.1),
                batch_size=1,
            ),
            replay_buffer=DiscreteContextualBanditReplayBuffer(1),
        )
        env = ContextualBanditLinearSyntheticEnvironment(
            action_space=action_space,
            observation_dim=observation_dim,
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

    def test_online_rl(self) -> None:
        env = FixedNumberOfStepsEnvironment(number_of_steps=100)
        agent = PearlAgent(TabularQLearning())
        online_learning(agent, env)

    def test_tabular_q_learning_online_rl(self) -> None:
        env = GymEnvironment("FrozenLake-v1", is_slippery=False)
        agent = PearlAgent(policy_learner=TabularQLearning())

        online_learning(agent, env, number_of_episodes=500)

        for _ in range(100):  # Should always reach the goal
            assert episode_return(agent, env, learn=False, exploit=True)[0] == 1.0

    def test_contextual_bandit_with_tabular_q_learning_online_rl(self) -> None:
        num_actions = 5
        max_action = num_actions - 1
        env = RewardIsEqualToTenTimesActionContextualBanditEnvironment(
            action_space=DiscreteActionSpace(
                actions=list(torch.arange(num_actions).view(-1, 1))
            )
        )
        # Because a contextual bandit environment is simply a regular Environment
        # with episodes lasting a single step, we can solve them with regular
        # RL algorithms such as tabular Q-learning.
        # This test ensures that is true (that even a non-CB method works with the CB environment).
        # In practice, CB-specific algorithms will be used.
        agent = PearlAgent(
            policy_learner=TabularQLearning(exploration_rate=0.1, learning_rate=0.1)
        )

        online_learning(agent, env, number_of_episodes=10000)

        # Should have learned to use action max_action with reward equal to max_action * 10
        for _ in range(100):
            assert (
                episode_return(agent, env, learn=False, exploit=True)[0]
                == max_action * 10
            )
