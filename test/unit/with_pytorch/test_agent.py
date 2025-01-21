# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


# pyre-strict


import time
import unittest
from typing import Callable, Generator, Tuple

import torch
from gym.envs.toy_text.frozen_lake import generate_random_map
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.api.agent import Agent
from pearl.api.environment import Environment
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.common.utils import init_weights
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    GaussianActorNetwork,
    VanillaActorNetwork,
    VanillaContinuousActorNetwork,
)

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    DuelingQValueNetwork,
    TwoTowerQValueNetwork,
    VanillaQValueMultiHeadNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (
    NormalDistributionExploration,
)

from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
)
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)

from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.deep_sarsa import DeepSARSA
from pearl.policy_learners.sequential_decision_making.double_dqn import DoubleDQN
from pearl.policy_learners.sequential_decision_making.implicit_q_learning import (
    ImplicitQLearning,
)
from pearl.policy_learners.sequential_decision_making.ppo import (
    PPOReplayBuffer,
    ProximalPolicyOptimization,
)
from pearl.policy_learners.sequential_decision_making.quantile_regression_deep_q_learning import (
    QuantileRegressionDeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.reinforce import (
    REINFORCE,
    REINFORCEReplayBuffer,
)
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import (
    SoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.tabular_q_learning import (
    TabularQLearning,
)
from pearl.policy_learners.sequential_decision_making.td3 import TD3
from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.sequential_decision_making.sarsa_replay_buffer import (
    SARSAReplayBuffer,
)
from pearl.safety_modules.risk_sensitive_safety_modules import (
    QuantileNetworkMeanVarianceSafetyModule,
)
from pearl.utils.functional_utils.train_and_eval.offline_learning_and_evaluation import (
    get_offline_data_in_buffer,
    offline_learning,
)
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
    OneHotObservationsFromDiscrete,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.environments.reward_is_equal_to_ten_times_action_multi_arm_bandit_environment import (  # Noqa E501
    RewardIsEqualToTenTimesActionMultiArmBanditEnvironment,
)
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


AgentAndEnvironmentMaker = Callable[[], Tuple[Agent, Environment]]


class TestAgentWithPyTorch(unittest.TestCase):
    def get_agent_makers(
        self,
    ) -> Generator[
        Tuple[str, AgentAndEnvironmentMaker, AgentAndEnvironmentMaker],
        None,
        None,
    ]:
        """
        Generator function that yields (str, function, function) tuples
        where the string is the name of agent type and the functions are
        AgentAndEnvironmentMaker functions that return a tuple of
        (agent, environment) for the given agent type.
        The first function returns a newly initialized agent and the second function
        returns a trained agent.
        This method is useful for programmatically iterating over
        the instances produced by this class for testing purposes.
        The initial motivation was for TestSerialization, where
        the first agent's state dict is serialized and then loaded
        into the second agent, and then the two agents are compared.

        IMPORTANT: the methods in this class producing agents must follow
        the naming pattern "get_new_<agent_type>_agent_and_environment" and
        "get_trained_<agent_type>_agent_and_environment" where <agent_type>
        is the description of the agent.
        """
        for method_name in dir(self):
            if method_name.startswith("get_new_") and method_name.endswith(
                "_agent_and_environment"
            ):
                # Extract the agent type
                agent_type = method_name.replace("get_new_", "").replace(
                    "_agent_and_environment", ""
                )

                # Construct the corresponding "get_trained_" method name
                trained_method_name = f"get_trained_{agent_type}_agent_and_environment"

                # Yield the method pair
                yield (
                    agent_type,
                    getattr(self, method_name),
                    getattr(self, trained_method_name),
                )

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

    def get_new_double_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n

        one_hot_action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        )

        observation_dim = env.observation_space.shape[0]
        action_dim = num_actions  # because we use one-hot action representation

        lstm_history_summarization_module = LSTMHistorySummarizationModule(
            observation_dim=observation_dim,
            action_dim=action_dim,
            history_length=5,  # Example history length
            hidden_dim=64,  # Example hidden dim
        )

        agent = PearlAgent(
            policy_learner=DoubleDQN(  # Use DoubleDQN here
                state_dim=observation_dim,
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=1,
                action_representation_module=one_hot_action_representation_module,
                exploration_module=EGreedyExploration(epsilon=0.1),
                history_summarization_module=lstm_history_summarization_module,
            ),
            replay_buffer=BasicReplayBuffer(10000),
        )
        return agent, env

    def get_trained_double_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_double_dqn_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_deep_sarsa_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        observation_dim = env.observation_space.shape[0]

        agent = PearlAgent(
            policy_learner=DeepSARSA(  # Use DeepSARSA here
                state_dim=observation_dim,
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=1,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
                exploration_module=EGreedyExploration(epsilon=0.1),
            ),
            replay_buffer=SARSAReplayBuffer(10000),
        )
        return agent, env

    def get_trained_deep_sarsa_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_deep_sarsa_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_iql_offline_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        IQLAgent = PearlAgent(
            policy_learner=ImplicitQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                exploration_module=NoExploration(),
                actor_hidden_dims=[64, 64],
                critic_hidden_dims=[64, 64],
                value_critic_hidden_dims=[64, 64],
                training_rounds=1,
                batch_size=32,
                expectile=0.70,
                temperature_advantage_weighted_regression=3.0,
                critic_soft_update_tau=0.005,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(200000),
        )
        return IQLAgent, env

    def get_trained_iql_offline_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        IQLAgent, env = self.get_new_iql_offline_agent_and_environment()
        # specify path for offline data set
        url = "https://raw.githubusercontent.com/jb3618columbia/offline_data/fbaccdd8d994479298c930d684ac49285f3cc901/offline_raw_transitions_dict_50k.pt"  # noqa: E501

        # get offline data from the specified path in a replay buffer
        is_action_continuous = False
        print(f"Loading offline data from {url}")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        offline_data_replay_buffer = get_offline_data_in_buffer(
            is_action_continuous=is_action_continuous,
            url=url,
            max_number_actions_if_discrete=num_actions,
        )

        # train conservative agent with offline data
        offline_learning(
            offline_agent=IQLAgent,
            data_buffer=offline_data_replay_buffer,
            number_of_batches=5,
            seed=100,
        )

        return IQLAgent, env

    def get_new_ppo_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                action_space=env.action_space,
                state_dim=env.observation_space.shape[0],
                actor_hidden_dims=[64, 64],
                critic_hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=32,
                epsilon=0.1,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=PPOReplayBuffer(10_000),
        )
        return agent, env

    def get_trained_ppo_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_ppo_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_ppo_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[
        PearlAgent,
        GymEnvironment,
    ]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        )

        # VanillaActorNetwork outputs a probability distribution over all actions,
        # so the output_dim is taken to be num_actions.
        actor_network_instance = VanillaActorNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[64, 64],
            output_dim=num_actions,
        )

        # PPO uses a VanillaValueNetwork by default
        critic_network_instance = VanillaValueNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[64, 64],
            output_dim=1,
        )

        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                training_rounds=20,
                batch_size=32,
                epsilon=0.1,
                action_representation_module=action_representation_module,
                actor_network_instance=actor_network_instance,
                critic_network_instance=critic_network_instance,
            ),
            replay_buffer=PPOReplayBuffer(10_000),
        )
        return agent, env

    def get_trained_ppo_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_ppo_with_network_instance_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_sac_agent_and_environment(self) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=SoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[64, 64, 64],
                critic_hidden_dims=[64, 64, 64],
                training_rounds=100,
                batch_size=100,
                entropy_coef=0.1,
                actor_learning_rate=0.0001,
                critic_learning_rate=0.0003,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(50000),
        )
        return agent, env

    def get_trained_sac_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_sac_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_sac_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        )

        # VanillaActorNetwork outputs a probability distribution over all actions,
        # so the output_dim is taken to be num_actions.
        actor_network = VanillaActorNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[64, 64, 64],
            output_dim=num_actions,
        )

        # we use twin critics of the type VanillaQValueNetwork.
        twin_critic_network = TwinCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=action_representation_module.max_number_actions,
            hidden_dims=[64, 64, 64],
            network_type=VanillaQValueNetwork,
            init_fn=init_weights,
        )

        agent = PearlAgent(
            policy_learner=SoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                training_rounds=100,
                batch_size=100,
                entropy_coef=0.1,
                actor_learning_rate=0.0001,
                critic_learning_rate=0.0003,
                action_representation_module=action_representation_module,
                actor_network_instance=actor_network,
                critic_network_instance=twin_critic_network,
            ),
            replay_buffer=BasicReplayBuffer(50000),
        )
        return agent, env

    def get_trained_sac_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_sac_with_network_instance_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_continuous_sac_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("Pendulum-v1")

        agent = PearlAgent(
            policy_learner=ContinuousSoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[64, 64],
                critic_hidden_dims=[64, 64],
                training_rounds=50,
                batch_size=100,
                entropy_coef=0.1,
                actor_learning_rate=0.001,
                critic_learning_rate=0.001,
            ),
            replay_buffer=BasicReplayBuffer(100000),
        )
        return agent, env

    def get_trained_continuous_sac_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_continuous_sac_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_continuous_sac_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("Pendulum-v1")

        # for continuous action spaces, Pearl currently only supports
        # IdentityActionRepresentationModule as the action representation module. So, the output_dim
        # argument of the GaussianActorNetwork is the same as the action space dimension. Also,
        # the action_dim argument for critic networks is the same as the action space dimension.
        actor_network_instance = GaussianActorNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[64, 64],
            output_dim=env.action_space.action_dim,
            action_space=env.action_space,
        )

        # SAC uses twin critics of the type VanillaQValueNetwork by default.
        twin_critic_network = TwinCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.action_dim,
            hidden_dims=[64, 64],
            network_type=VanillaQValueNetwork,
            init_fn=init_weights,
        )

        agent = PearlAgent(
            policy_learner=ContinuousSoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                training_rounds=50,
                batch_size=100,
                entropy_coef=0.1,
                actor_learning_rate=0.001,
                critic_learning_rate=0.001,
                actor_network_instance=actor_network_instance,
                critic_network_instance=twin_critic_network,
            ),
            replay_buffer=BasicReplayBuffer(100000),
        )
        return agent, env

    def get_trained_continuous_sac_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = (
            self.get_new_continuous_sac_with_network_instance_agent_and_environment()
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_cql_online_agent_and_environment(
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
            replay_buffer=BasicReplayBuffer(10_000),
        )
        return agent, env

    def get_trained_cql_online_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_cql_online_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_cql_offline_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        conservativeDQN_agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=100,
                is_conservative=True,
                conservative_alpha=4.0,
                batch_size=128,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=BasicReplayBuffer(10000),
        )
        return conservativeDQN_agent, env

    def get_trained_cql_offline_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        conservativeDQN_agent, env = self.get_new_cql_offline_agent_and_environment()

        # specify path for offline data set
        url = "https://raw.githubusercontent.com/jb3618columbia/offline_data/ee11452e5c6116d12cd3c1cab25aff39ad7d6ebf/offline_raw_transitions_dict_50k.pt"  # noqa: E501

        # get offline data from the specified path in a replay buffer
        is_action_continuous = False
        print(f"Loading offline data from {url}")

        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n

        offline_data_replay_buffer = get_offline_data_in_buffer(
            is_action_continuous=is_action_continuous,
            url=url,
            max_number_actions_if_discrete=num_actions,
        )

        # train conservative agent with offline data
        print("offline data in replay buffer; start offline training")
        offline_learning(
            offline_agent=conservativeDQN_agent,
            data_buffer=offline_data_replay_buffer,
            number_of_batches=20,
            seed=100,
        )
        return conservativeDQN_agent, env

    def get_new_ddpg_agent_and_environment(self) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("Pendulum-v1")
        agent = PearlAgent(
            policy_learner=DeepDeterministicPolicyGradient(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[400, 300],
                critic_hidden_dims=[400, 300],
                critic_learning_rate=1e-3,
                actor_learning_rate=1e-3,
                training_rounds=5,
                actor_soft_update_tau=0.05,
                critic_soft_update_tau=0.05,
                exploration_module=NormalDistributionExploration(
                    mean=0,
                    std_dev=0.2,
                ),
            ),
            replay_buffer=BasicReplayBuffer(50000),
        )
        return agent, env

    def get_trained_ddpg_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_ddpg_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_td3_agent_and_environment(self) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("Pendulum-v1")
        agent = PearlAgent(
            policy_learner=TD3(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[400, 300],
                critic_hidden_dims=[400, 300],
                critic_learning_rate=1e-2,
                actor_learning_rate=1e-3,
                training_rounds=5,
                actor_soft_update_tau=0.05,
                critic_soft_update_tau=0.05,
                exploration_module=NormalDistributionExploration(
                    mean=0,
                    std_dev=0.2,
                ),
            ),
            replay_buffer=BasicReplayBuffer(50000),
        )
        return agent, env

    def get_trained_td3_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_td3_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_td3_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("Pendulum-v1")

        # Note: for continuous action spaces, Pearl currently only supports
        # IdentityActionRepresentationModule as the action representation module. So, the output_dim
        # argument of the VanillaContinuousActorNetwork is the same as the action space dimension.
        # For this reason, the action_dim argument for critic networks is the same as the action
        # space dimension.

        # td3 uses a deterministic policy network (e.g. VanillaContinuousActorNetwork) by default.
        actor_network_instance = VanillaContinuousActorNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[400, 300],
            output_dim=env.action_space.action_dim,
            action_space=env.action_space,
        )

        twin_critic_network = TwinCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.action_dim,
            hidden_dims=[400, 300],
            network_type=VanillaQValueNetwork,
            init_fn=init_weights,
        )

        agent = PearlAgent(
            policy_learner=TD3(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[400, 300],
                critic_hidden_dims=[400, 300],
                critic_learning_rate=1e-2,
                actor_learning_rate=1e-3,
                training_rounds=5,
                actor_soft_update_tau=0.05,
                critic_soft_update_tau=0.05,
                exploration_module=NormalDistributionExploration(
                    mean=0,
                    std_dev=0.2,
                ),
                actor_network_instance=actor_network_instance,
                critic_network_instance=twin_critic_network,
            ),
            replay_buffer=BasicReplayBuffer(50000),
        )
        return agent, env

    def get_trained_td3_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_td3_with_network_instance_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_dqn_multihead_agent_and_environment(
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
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
                network_instance=VanillaQValueMultiHeadNetwork(
                    state_dim=env.observation_space.shape[0],
                    action_dim=num_actions,
                    hidden_dims=[64, 64],
                    output_dim=num_actions,
                ),
            ),
            replay_buffer=BasicReplayBuffer(10_000),
        )
        return agent, env

    def get_trained_dqn_multihead_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_dqn_multihead_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_dqn_on_frozen_lake_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, Environment]:
        env = OneHotObservationsFromDiscrete(
            GymEnvironment(
                "FrozenLake-v1", is_slippery=False, desc=generate_random_map(size=4)
            )
        )
        assert isinstance(env.action_space, DiscreteSpace)
        state_dim = env.observation_space.n
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=state_dim,
                action_space=env.action_space,
                hidden_dims=[64],
                training_rounds=20,
            ),
            replay_buffer=BasicReplayBuffer(1000),
        )
        return agent, env

    def get_trained_dqn_on_frozen_lake_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, Environment]:
        agent, env = self.get_new_dqn_on_frozen_lake_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_reinforce_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=REINFORCE(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[64, 64],
                use_critic=True,
                critic_hidden_dims=[64, 64],
                training_rounds=8,
                batch_size=64,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            replay_buffer=REINFORCEReplayBuffer(10_000),
        )
        return agent, env

    def get_trained_reinforce_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_reinforce_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_reinforce_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        )

        # VanillaActorNetwork outputs a probability distribution over all actions,
        # so the output_dim is taken to be num_actions.
        actor_network_instance = VanillaActorNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[64, 64],
            output_dim=num_actions,
        )

        # REINFORCE uses a VanillaValueNetwork by default
        critic_network_instance = VanillaValueNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dims=[64, 64],
            output_dim=1,
        )

        agent = PearlAgent(
            policy_learner=REINFORCE(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                use_critic=True,
                training_rounds=8,
                batch_size=64,
                action_representation_module=action_representation_module,
                actor_network_instance=actor_network_instance,
                critic_network_instance=critic_network_instance,
            ),
            replay_buffer=REINFORCEReplayBuffer(10_000),
        )

        return agent, env

    def get_trained_reinforce_with_network_instance_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = (
            self.get_new_reinforce_with_network_instance_agent_and_environment()
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    def get_new_qr_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n
        agent = PearlAgent(
            policy_learner=QuantileRegressionDeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64, 64],
                exploration_module=EGreedyExploration(0.10),
                learning_rate=5e-4,
                training_rounds=20,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=num_actions
                ),
            ),
            safety_module=QuantileNetworkMeanVarianceSafetyModule(0.2),
            replay_buffer=BasicReplayBuffer(10_000),
        )
        return agent, env

    def get_trained_qr_dqn_agent_and_environment(
        self,
    ) -> Tuple[PearlAgent, GymEnvironment]:
        agent, env = self.get_new_qr_dqn_agent_and_environment()
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )
        return agent, env

    """
    From here on, we test the constructions of all agent types defined above,
    and, when possible, learning success.
    """

    def test_construction_of_all_types(self) -> None:
        for (
            agent_type,
            _new_agent_function,
            trained_agent_function,
        ) in self.get_agent_makers():
            print(f"Testing construction of {agent_type}")
            start = time.time()
            trained_agent_function()
            end = time.time()
            print(
                f"Completed construction of {agent_type} in {end - start:.2f} seconds"
            )

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
