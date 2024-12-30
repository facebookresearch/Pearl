# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


# pyre-strict


import unittest
from typing import Tuple

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
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
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import (
    online_learning,
    online_learning_to_png_graph,
    run_episode,
)

from pearl.utils.instantiations.environments.contextual_bandit_linear_synthetic_environment import (
    ContextualBanditLinearSyntheticEnvironment,
)
from pearl.utils.instantiations.environments.environments import (
    FixedNumberOfStepsEnvironment,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.environments.reward_is_equal_to_ten_times_action_multi_arm_bandit_environment import (  # Noqa E501
    RewardIsEqualToTenTimesActionMultiArmBanditEnvironment,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestAgentWithPyTorch(unittest.TestCase):
    def get_new_dqn_agent_and_environment(self) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=1,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(10000),
        )
        return agent, env

    def get_trained_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_dqn_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_conservative_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                is_conservative=True,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(10000),
        )
        return agent, env

    def get_trained_conservative_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_conservative_dqn_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_dqn_dueling_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                hidden_dims=[64, 64],
                action_space=env.action_space,
                training_rounds=20,
                network_type=DuelingQValueNetwork,
                batch_size=128,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(10000),
        )
        return agent, env

    def get_trained_dqn_dueling_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_dqn_dueling_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_dqn_two_tower_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
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
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(10000),
        )
        return agent, env

    def get_trained_dqn_two_tower_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_dqn_two_tower_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_linear_contextual_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, ContextualBanditLinearSyntheticEnvironment]:
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
            replay_buffer=BasicReplayBuffer(1),
        )
        env = ContextualBanditLinearSyntheticEnvironment(
            action_space=action_space,
            observation_dim=observation_dim,
        )
        return agent, env

    def get_trained_linear_contextual_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, ContextualBanditLinearSyntheticEnvironment]:
        agent, env = self.get_new_linear_contextual_agent_and_environment()

        regrets = []
        for _ in range(100):
            observation, action_space = env.reset()
            agent.reset(observation, action_space)
            action = agent.act()
            regret = env.get_regret(action)
            action_result = env.step(action)
            agent.observe(action_result)
            agent.learn()
            assert isinstance(regret, torch.Tensor)
            regrets.append(regret.squeeze().item())

        # to test learning ability of linear contextual bandits we check
        # that the regret is decreasing over learning steps
        self.assertTrue(sum(regrets[10:]) >= sum(regrets[-10:]))

        return agent, env

    def get_new_online_rl_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, FixedNumberOfStepsEnvironment]:
        env = FixedNumberOfStepsEnvironment(max_number_of_steps=100)
        agent = PearlAgent(TabularQLearning())
        return agent, env

    def get_trained_online_rl_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, FixedNumberOfStepsEnvironment]:
        agent, env = self.get_new_online_rl_agent_and_environment()
        online_learning(agent, env, number_of_episodes=1000)
        return agent, env

    def get_new_tabular_q_learning_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("FrozenLake-v1", is_slippery=False)
        agent = PearlAgent(policy_learner=TabularQLearning(exploration_rate=0.7))
        return agent, env

    def get_trained_tabular_q_learning_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_tabular_q_learning_agent_and_environment()
        # We use a large exploration rate because the exploitation action
        # is always the first one among those with the highest value
        # (so that the agent is deterministic in the absence of exploration).
        # For FrozenLake, this results in action 0 which is "moving left"
        # and has no effect for the initial position in the left top corner.
        # Ideally, we should use a smarter exploration strategy that
        # picks an action randomly but favors the best ones
        # (propensity exploration).
        # For FrozenLake, especially at the beginning of training,
        # this would result in a random choice between the
        # initially equally valueless actions, resulting in effective
        # exploration, but focusing more on valuable actions
        # as training progresses.
        # TODO: modify tabular Q-learning so it accepts
        # a greater variety of exploration modules.

        online_learning(agent, env, number_of_episodes=6000)

        return agent, env

    def get_new_contextual_bandit_with_tabular_q_learning_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, RewardIsEqualToTenTimesActionMultiArmBanditEnvironment]:
        num_actions = 5
        env = RewardIsEqualToTenTimesActionMultiArmBanditEnvironment(
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
        return agent, env

    def get_trained_contextual_bandit_with_tabular_q_learning_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, RewardIsEqualToTenTimesActionMultiArmBanditEnvironment]:
        agent, env = (
            self.get_new_contextual_bandit_with_tabular_q_learning_agent_and_environment()  # noqa E501
        )

        online_learning(agent, env, number_of_episodes=10000)
        return agent, env

    """
    A collection of Agent tests using PyTorch (this saves around 100 secs in test loading).
    For tests not involving PyTorch, use see test/without_pytorch.
    """

    def test_deep_td_learning_online_rl_sanity_check(self) -> None:
        self.get_trained_dqn_agent_and_environment()

    def test_conservative_deep_td_learning_online_rl_sanity_check(self) -> None:
        self.get_trained_conservative_dqn_agent_and_environment()

    def test_deep_td_learning_online_rl_sanity_check_dueling(
        self,
    ) -> None:
        self.get_trained_dqn_dueling_agent_and_environment()

    def test_deep_td_learning_online_rl_two_tower_network(self) -> None:
        self.get_trained_dqn_two_tower_agent_and_environment()

    def test_with_linear_contextual(self) -> None:
        self.get_trained_linear_contextual_agent_and_environment()

    def test_online_rl(self) -> None:
        self.get_trained_online_rl_agent_and_environment()

    def test_tabular_q_learning_online_rl(self) -> None:
        agent, env = self.get_trained_tabular_q_learning_agent_and_environment()

        for _ in range(100):  # Should always reach the goal
            episode_info, total_steps = run_episode(
                agent, env, learn=False, exploit=True
            )
            assert episode_info["return"] == 1.0

    def test_contextual_bandit_with_tabular_q_learning_online_rl(self) -> None:
        agent, env = (
            self.get_trained_contextual_bandit_with_tabular_q_learning_agent_and_environment()
        )
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        max_action = num_actions - 1

        # Should have learned to use action max_action with reward equal to max_action * 10
        for _ in range(100):
            episode_info, total_steps = run_episode(
                agent, env, learn=False, exploit=True
            )
            assert episode_info["return"] == max_action * 10
