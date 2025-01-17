# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import List

import torch
from later.unittest import TestCase
from pearl.action_representation_modules.binary_action_representation_module import (
    BinaryActionTensorRepresentationModule,
)
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.history_summarization_modules.stacking_history_summarization_module import (
    StackingHistorySummarizationModule,
)
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.neural_networks.contextual_bandit.neural_linear_regression import (
    NeuralLinearRegression,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    EnsembleQValueNetwork,
)
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
)
from pearl.policy_learners.contextual_bandits.disjoint_bandit import (
    DisjointBanditContainer,
)
from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import (
    NeuralLinearBandit,
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
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import (
    FastCBExploration,
    SquareCBExploration,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (  # noqa E501
    ThompsonSamplingExplorationLinear,
    ThompsonSamplingExplorationLinearDisjoint,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
    UCBExploration,
    VanillaUCBExploration,
)
from pearl.policy_learners.exploration_modules.sequential_decision_making.deep_exploration import (
    DeepExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.sequential_decision_making.bootstrapped_dqn import (
    BootstrappedDQN,
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
    ProximalPolicyOptimization,
)
from pearl.policy_learners.sequential_decision_making.quantile_regression_deep_q_learning import (
    QuantileRegressionDeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.reinforce import REINFORCE
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
from pearl.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.safety_modules.reward_constrained_safety_module import (
    RCSafetyModuleCostCriticContinuousAction,
)
from pearl.safety_modules.risk_sensitive_safety_modules import (
    QuantileNetworkMeanVarianceSafetyModule,
    RiskNeutralSafetyModule,
)
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import nn


class TestCompare(TestCase):
    """
    A suite of tests for `compare` methods of Pearl classes
    """

    def test_compare_lstm_history_summarization_module(self) -> None:
        module1 = LSTMHistorySummarizationModule(
            history_length=10,
            hidden_dim=32,
            num_layers=2,
            observation_dim=6,
            action_dim=4,
        )
        module2 = LSTMHistorySummarizationModule(
            history_length=10,
            hidden_dim=32,
            num_layers=2,
            observation_dim=6,
            action_dim=4,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should be different due to random LSTM init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the LSTMs have the same weights
        for param1, param2 in zip(module1.lstm.parameters(), module2.lstm.parameters()):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_stacking_history_summarization_module(self) -> None:
        module1 = StackingHistorySummarizationModule(
            observation_dim=6, action_dim=4, history_length=10
        )
        module2 = StackingHistorySummarizationModule(
            observation_dim=6, action_dim=4, history_length=10
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.history_length = 12

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_identity_history_summarization_module(self) -> None:
        module1 = IdentityHistorySummarizationModule()
        module2 = IdentityHistorySummarizationModule()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

    def test_compare_linear_regression(self) -> None:
        module1 = LinearRegression(feature_dim=10, l2_reg_lambda=0.1, gamma=0.95)
        module2 = LinearRegression(feature_dim=10, l2_reg_lambda=0.1, gamma=0.95)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.gamma = 0.9

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Make some data and learn it
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)
        module1.learn_batch(x, y)

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Set gamma to the same value as module1's and learn the same data
        module2.gamma = module1.gamma
        module2.learn_batch(x, y)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_neural_linear_regression(self) -> None:
        module1 = NeuralLinearRegression(
            feature_dim=10, hidden_dims=[32, 16], l2_reg_lambda_linear=0.1, gamma=0.95
        )
        module2 = NeuralLinearRegression(
            feature_dim=10, hidden_dims=[32, 16], l2_reg_lambda_linear=0.1, gamma=0.95
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have random differences initially)
        self.assertNotEqual(module1.compare(module2), "")

        # Load state dict from one to the other and compare again
        module1.load_state_dict(module2.state_dict())
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.nn_e2e = not module2.nn_e2e

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Undo flip, load state dict from one to the other and compare again
        module2.nn_e2e = not module2.nn_e2e
        module1.load_state_dict(module2.state_dict())
        self.assertEqual(module1.compare(module2), "")

    def test_compare_egreedy_exploration(self) -> None:
        module1 = EGreedyExploration(
            epsilon=0.1, start_epsilon=0.9, end_epsilon=0.05, warmup_steps=1000
        )
        module2 = EGreedyExploration(
            epsilon=0.1, start_epsilon=0.9, end_epsilon=0.05, warmup_steps=1000
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.end_epsilon = 0.1

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_no_exploration(self) -> None:
        module1 = NoExploration()
        module2 = NoExploration()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

    def test_compare_normal_distribution_exploration(self) -> None:
        module1 = NormalDistributionExploration(mean=0.0, std_dev=1.0)
        module2 = NormalDistributionExploration(mean=0.0, std_dev=1.0)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._std_dev = 0.5

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_propensity_exploration(self) -> None:
        module1 = PropensityExploration()
        module2 = PropensityExploration()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

    def test_compare_square_cb_exploration(self) -> None:
        module1 = SquareCBExploration(gamma=0.5, reward_lb=-1.0, reward_ub=1.0)
        module2 = SquareCBExploration(gamma=0.5, reward_lb=-1.0, reward_ub=1.0)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.reward_ub = 2.0

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_fast_cb_exploration(self) -> None:
        module1 = FastCBExploration(gamma=0.5, reward_lb=-1.0, reward_ub=1.0)
        module2 = FastCBExploration(gamma=0.5, reward_lb=-1.0, reward_ub=1.0)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.reward_lb = -2.0

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_ucb_exploration(self) -> None:
        module1 = UCBExploration(alpha=1.0)
        module2 = UCBExploration(alpha=1.0)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._alpha = 0.5

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_disjoint_ucb_exploration(self) -> None:
        module1 = DisjointUCBExploration(alpha=1.0)
        module2 = DisjointUCBExploration(alpha=1.0)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._alpha = 0.5

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_vanilla_ucb_exploration(self) -> None:
        module1 = VanillaUCBExploration()
        module2 = VanillaUCBExploration()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.action_execution_count = {0: 10, 1: 5}

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_deep_exploration(self) -> None:
        # Create q_ensemble_networks with the same parameters
        q_ensemble_network1 = EnsembleQValueNetwork(
            state_dim=10,
            action_dim=4,
            hidden_dims=[32, 16],  # Example hidden dims
            output_dim=1,  # Example output dim
            ensemble_size=5,
        )
        q_ensemble_network2 = EnsembleQValueNetwork(
            state_dim=10,
            action_dim=4,
            hidden_dims=[32, 16],  # Same hidden dims
            output_dim=1,  # Same output dim
            ensemble_size=5,
        )

        # Create action_representation_modules
        action_representation_module1 = nn.Linear(4, 16)
        action_representation_module2 = nn.Linear(4, 16)

        module1 = DeepExploration(
            q_ensemble_network=q_ensemble_network1,
            action_representation_module=action_representation_module1,
        )
        module2 = DeepExploration(
            q_ensemble_network=q_ensemble_network2,
            action_representation_module=action_representation_module2,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have differences initially)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the q_ensemble_networks have the same weights
        for param1, param2 in zip(
            q_ensemble_network1._model.parameters(),
            q_ensemble_network2._model.parameters(),
        ):
            param2.data.copy_(param1.data)

        # Make the action_representation_modules have the same weights
        for param1, param2 in zip(
            module1.action_representation_module.parameters(),
            module2.action_representation_module.parameters(),
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_thompson_sampling_exploration_linear(self) -> None:
        module1 = ThompsonSamplingExplorationLinear(enable_efficient_sampling=True)
        module2 = ThompsonSamplingExplorationLinear(enable_efficient_sampling=True)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._enable_efficient_sampling = False

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_thompson_sampling_exploration_linear_disjoint(self) -> None:
        module1 = ThompsonSamplingExplorationLinearDisjoint(
            enable_efficient_sampling=True
        )
        module2 = ThompsonSamplingExplorationLinearDisjoint(
            enable_efficient_sampling=True
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._enable_efficient_sampling = False

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_neural_linear_bandit(self) -> None:
        # Create exploration modules (e.g., UCBExploration)
        exploration_module1 = UCBExploration(alpha=1.0)
        exploration_module2 = UCBExploration(alpha=1.0)

        # Initialize with the same random seed for consistent initialization
        torch.manual_seed(0)
        module1 = NeuralLinearBandit(
            feature_dim=10,
            hidden_dims=[32, 16],
            exploration_module=exploration_module1,
            loss_type="mse",
            apply_discounting_interval=100,
        )
        torch.manual_seed(0)  # Reset the seed for the second module
        module2 = NeuralLinearBandit(
            feature_dim=10,
            hidden_dims=[32, 16],
            exploration_module=exploration_module2,
            loss_type="mse",
            apply_discounting_interval=100,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences now)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.loss_type = "mae"

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_warmup(self) -> None:
        # Create some exploration module (e.g., EGreedyExploration)
        base_exploration_module1 = EGreedyExploration(epsilon=0.1)
        base_exploration_module2 = EGreedyExploration(epsilon=0.1)

        module1 = Warmup(exploration_module=base_exploration_module1, warmup_steps=1000)
        module2 = Warmup(exploration_module=base_exploration_module2, warmup_steps=1000)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.warmup_steps = 500

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_one_hot_action_tensor_representation_module(self) -> None:
        module1 = OneHotActionTensorRepresentationModule(max_number_actions=4)
        module2 = OneHotActionTensorRepresentationModule(max_number_actions=4)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._max_number_actions = 5

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_binary_action_tensor_representation_module(self) -> None:
        module1 = BinaryActionTensorRepresentationModule(bits_num=3)
        module2 = BinaryActionTensorRepresentationModule(bits_num=3)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._bits_num = 4

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_identity_action_representation_module(self) -> None:
        module1 = IdentityActionRepresentationModule(
            max_number_actions=4, representation_dim=2
        )
        module2 = IdentityActionRepresentationModule(
            max_number_actions=4, representation_dim=2
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._max_number_actions = 5

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_linear_bandit(self) -> None:
        # Create exploration modules (e.g., UCBExploration)
        exploration_module1 = UCBExploration(alpha=1.0)
        exploration_module2 = UCBExploration(alpha=1.0)

        module1 = LinearBandit(
            feature_dim=10,
            exploration_module=exploration_module1,
            apply_discounting_interval=100,
        )
        module2 = LinearBandit(
            feature_dim=10,
            exploration_module=exploration_module2,
            apply_discounting_interval=100,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.apply_discounting_interval = 200

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_disjoint_bandit_container(self) -> None:
        arm_bandits1: List[ContextualBanditBase] = [
            LinearBandit(
                feature_dim=10,
                l2_reg_lambda=0.1,
                gamma=0.95,
                exploration_module=UCBExploration(alpha=1.0),  # Add exploration module
            )
            for _ in range(3)
        ]
        arm_bandits2: List[ContextualBanditBase] = [
            LinearBandit(
                feature_dim=10,
                l2_reg_lambda=0.1,
                gamma=0.95,
                exploration_module=UCBExploration(alpha=1.0),  # Add exploration module
            )
            for _ in range(3)
        ]

        # Create exploration modules (using a dummy one for this example)
        exploration_module1 = NoExploration()
        exploration_module2 = NoExploration()

        module1 = DisjointBanditContainer(
            feature_dim=10,
            arm_bandits=arm_bandits1,
            exploration_module=exploration_module1,
        )
        module2 = DisjointBanditContainer(
            feature_dim=10,
            arm_bandits=arm_bandits2,
            exploration_module=exploration_module2,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._state_features_only = not module2._state_features_only

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Modify an arm bandit in module2
        assert isinstance((ab0 := arm_bandits2[0]), LinearBandit)
        ab0.model.gamma = 0.9

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_disjoint_linear_bandit(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace([torch.tensor(i) for i in range(3)])
        action_space2 = DiscreteActionSpace([torch.tensor(i) for i in range(3)])

        # Create exploration modules (e.g., UCBExploration)
        exploration_module1 = UCBExploration(alpha=1.0)
        exploration_module2 = UCBExploration(alpha=1.0)

        module1 = DisjointLinearBandit(
            feature_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
        )
        module2 = DisjointLinearBandit(
            feature_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._state_features_only = not module2._state_features_only

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Modify a linear regression in module2
        module2_lr_0 = module2._linear_regressions_list[0]
        assert isinstance(module2_lr_0, LinearRegression)
        module2_lr_0.gamma = 0.9

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_neural_bandit(self) -> None:
        # Create exploration modules (e.g., UCBExploration)
        exploration_module1 = UCBExploration(alpha=1.0)
        exploration_module2 = UCBExploration(alpha=1.0)

        module1 = NeuralBandit(
            feature_dim=10,
            hidden_dims=[32, 16],
            exploration_module=exploration_module1,
            loss_type="mse",
        )
        module2 = NeuralBandit(
            feature_dim=10,
            hidden_dims=[32, 16],
            exploration_module=exploration_module2,
            loss_type="mse",
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have differences due to random initialization)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the neural networks have the same weights
        for param1, param2 in zip(
            module1.model.parameters(), module2.model.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.loss_type = "mae"

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_identity_safety_module(self) -> None:
        module1 = IdentitySafetyModule()
        module2 = IdentitySafetyModule()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

    def test_compare_rc_safety_module_cost_critic_continuous_action(self) -> None:
        action_space1 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )
        action_space2 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )

        module1 = RCSafetyModuleCostCriticContinuousAction(
            constraint_value=1.0,
            state_dim=10,
            action_space=action_space1,
            critic_hidden_dims=[32, 16],
        )
        module2 = RCSafetyModuleCostCriticContinuousAction(
            constraint_value=1.0,
            state_dim=10,
            action_space=action_space2,
            critic_hidden_dims=[32, 16],
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2
        # (should have differences initially due to random initialization)
        self.assertNotEqual(module1.compare(module2), "")

        # Make them equal
        module1.cost_critic.load_state_dict(module2.cost_critic.state_dict())
        module1.target_of_cost_critic.load_state_dict(
            module2.target_of_cost_critic.state_dict()
        )

        # Should now be equal
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.constraint_value = 2.0

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_risk_neutral_safety_module(self) -> None:
        module1 = RiskNeutralSafetyModule()
        module2 = RiskNeutralSafetyModule()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

    def test_compare_quantile_network_mean_variance_safety_module(self) -> None:
        module1 = QuantileNetworkMeanVarianceSafetyModule(
            variance_weighting_coefficient=0.1
        )
        module2 = QuantileNetworkMeanVarianceSafetyModule(
            variance_weighting_coefficient=0.1
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2._beta = 0.2

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_deep_q_learning(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = EGreedyExploration(epsilon=0.1)
        exploration_module2 = EGreedyExploration(epsilon=0.1)

        module1 = DeepQLearning(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            discount_factor=0.99,
        )
        module2 = DeepQLearning(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random Q-net init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(module1._Q.parameters(), module2._Q.parameters()):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._Q_target.parameters(), module2._Q_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_double_dqn(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = EGreedyExploration(epsilon=0.1)
        exploration_module2 = EGreedyExploration(epsilon=0.1)

        module1 = DoubleDQN(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            discount_factor=0.99,
        )
        module2 = DoubleDQN(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random Q-net init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(module1._Q.parameters(), module2._Q.parameters()):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._Q_target.parameters(), module2._Q_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_bootstrapped_dqn(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create q_ensemble_networks
        q_ensemble_network1 = EnsembleQValueNetwork(
            state_dim=10,
            action_dim=4,
            hidden_dims=[32, 16],
            output_dim=1,
            ensemble_size=5,
        )
        q_ensemble_network2 = EnsembleQValueNetwork(
            state_dim=10,
            action_dim=4,
            hidden_dims=[32, 16],
            output_dim=1,
            ensemble_size=5,
        )

        module1 = BootstrappedDQN(
            action_space=action_space1,
            q_ensemble_network=q_ensemble_network1,
            discount_factor=0.99,
        )
        module2 = BootstrappedDQN(
            action_space=action_space2,
            q_ensemble_network=q_ensemble_network2,
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random Q-net init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(module1._Q.parameters(), module2._Q.parameters()):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._Q_target.parameters(), module2._Q_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_deep_sarsa(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = EGreedyExploration(epsilon=0.1)
        exploration_module2 = EGreedyExploration(epsilon=0.1)

        module1 = DeepSARSA(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            discount_factor=0.99,
        )
        module2 = DeepSARSA(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random Q-net init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(module1._Q.parameters(), module2._Q.parameters()):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._Q_target.parameters(), module2._Q_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_tabular_q_learning(self) -> None:
        module1 = TabularQLearning(
            learning_rate=0.1, discount_factor=0.9, exploration_rate=0.05
        )
        module2 = TabularQLearning(
            learning_rate=0.1, discount_factor=0.9, exploration_rate=0.05
        )

        # Set the same action space for both modules
        action_space = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        module1.reset(action_space)
        module2.reset(action_space)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.learning_rate = 0.2

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Update q-values in module1
        state = torch.tensor(0)  # Example state
        action = 0  # Example action
        module1.q_values[(state, action)] = 1.0

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_deep_deterministic_policy_gradient(self) -> None:
        # Create action spaces
        action_space1 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )
        action_space2 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )

        # Create exploration modules
        exploration_module1 = NormalDistributionExploration(mean=0.0, std_dev=0.1)
        exploration_module2 = NormalDistributionExploration(mean=0.0, std_dev=0.1)

        module1 = DeepDeterministicPolicyGradient(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            discount_factor=0.99,
        )
        module2 = DeepDeterministicPolicyGradient(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._actor_target.parameters(), module2._actor_target.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic_target.parameters(), module2._critic_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_implicit_q_learning(self) -> None:
        # Create action spaces
        action_space1 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )
        action_space2 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )

        # Create exploration modules
        exploration_module1 = NoExploration()
        exploration_module2 = NoExploration()

        module1 = ImplicitQLearning(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            value_critic_hidden_dims=[32, 16],
            expectile=0.7,
        )
        module2 = ImplicitQLearning(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            value_critic_hidden_dims=[32, 16],
            expectile=0.7,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._value_network.parameters(), module2._value_network.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic_target.parameters(), module2._critic_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_proximal_policy_optimization(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = PropensityExploration()
        exploration_module2 = PropensityExploration()

        module1 = ProximalPolicyOptimization(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            discount_factor=0.99,
            epsilon=0.2,
            trace_decay_param=0.95,
            entropy_bonus_scaling=0.01,
        )
        module2 = ProximalPolicyOptimization(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            discount_factor=0.99,
            epsilon=0.2,
            trace_decay_param=0.95,
            entropy_bonus_scaling=0.01,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_reinforce(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = PropensityExploration()
        exploration_module2 = PropensityExploration()

        module1 = REINFORCE(
            state_dim=10,
            use_critic=True,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],  # Even if not used, include for comparison
            discount_factor=0.99,
        )
        module2 = REINFORCE(
            state_dim=10,
            use_critic=True,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],  # Even if not used, include for comparison
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        # Even if the critic is not used, synchronize its weights for the comparison
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_continuous_soft_actor_critic(self) -> None:
        # Create action spaces
        action_space1 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )
        action_space2 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )

        # Create exploration modules
        exploration_module1 = NoExploration()
        exploration_module2 = NoExploration()

        module1 = ContinuousSoftActorCritic(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            entropy_autotune=True,  # Enable entropy autotuning
        )
        module2 = ContinuousSoftActorCritic(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            entropy_autotune=True,  # Enable entropy autotuning
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic_target.parameters(), module2._critic_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_soft_actor_critic(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = PropensityExploration()
        exploration_module2 = PropensityExploration()

        module1 = SoftActorCritic(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            entropy_autotune=True,  # Enable entropy autotuning
        )
        module2 = SoftActorCritic(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            entropy_autotune=True,  # Enable entropy autotuning
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic_target.parameters(), module2._critic_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_td3(self) -> None:
        # Create action spaces
        action_space1 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )
        action_space2 = BoxActionSpace(
            low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0])
        )

        # Create exploration modules
        exploration_module1 = NormalDistributionExploration(mean=0.0, std_dev=0.1)
        exploration_module2 = NormalDistributionExploration(mean=0.0, std_dev=0.1)

        module1 = TD3(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            actor_update_freq=2,
            actor_update_noise=0.2,
            actor_update_noise_clip=0.5,
        )
        module2 = TD3(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            actor_update_freq=2,
            actor_update_noise=0.2,
            actor_update_noise_clip=0.5,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(
            module1._actor.parameters(), module2._actor.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic.parameters(), module2._critic.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._actor_target.parameters(), module2._actor_target.parameters()
        ):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._critic_target.parameters(), module2._critic_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_quantile_regression_deep_q_learning(self) -> None:
        # Create action spaces
        action_space1 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))
        action_space2 = DiscreteActionSpace(actions=list(torch.arange(4).view(-1, 1)))

        # Create exploration modules
        exploration_module1 = EGreedyExploration(epsilon=0.1)
        exploration_module2 = EGreedyExploration(epsilon=0.1)

        module1 = QuantileRegressionDeepQLearning(
            state_dim=10,
            action_space=action_space1,
            exploration_module=exploration_module1,
            hidden_dims=[32, 16],
            num_quantiles=20,  # Example number of quantiles
            learning_rate=0.001,
            discount_factor=0.99,
        )
        module2 = QuantileRegressionDeepQLearning(
            state_dim=10,
            action_space=action_space2,
            exploration_module=exploration_module2,
            hidden_dims=[32, 16],
            num_quantiles=20,  # Same number of quantiles
            learning_rate=0.001,
            discount_factor=0.99,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (might have differences initially due to random Q-net init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the modules have the same weights
        for param1, param2 in zip(module1._Q.parameters(), module2._Q.parameters()):
            param2.data.copy_(param1.data)
        for param1, param2 in zip(
            module1._Q_target.parameters(), module2._Q_target.parameters()
        ):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")
