# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

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
