# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from copy import deepcopy

import gymnasium as gym
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)

from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.common.value_networks import (
    DuelingQValueNetwork,
    EnsembleQValueNetwork,
    VanillaQValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    DynamicActionActorNetwork,
    GaussianActorNetwork,
    VanillaActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (  # noqa E501
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (  # noqa E501
    NormalDistributionExploration,
)
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
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
from pearl.policy_learners.sequential_decision_making.quantile_regression_deep_q_learning import (  # noqa E501
    QuantileRegressionDeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.reinforce import REINFORCE
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import (
    SoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.td3 import TD3
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import (
    BootstrapReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (  # noqa E501
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.fifo_on_policy_replay_buffer import (  # noqa E501
    FIFOOnPolicyReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.on_policy_episodic_replay_buffer import (  # noqa E501
    OnPolicyEpisodicReplayBuffer,
)
from pearl.safety_modules.reward_constrained_safety_module import (
    RCSafetyModuleCostCriticContinuousAction,
)
from pearl.safety_modules.risk_sensitive_safety_modules import (
    QuantileNetworkMeanVarianceSafetyModule,
)

from pearl.user_envs import (
    AcrobotPartialObservableWrapper,
    AcrobotSparseRewardWrapper,
    CartPolePartialObservableWrapper,
    GymAvgTorqueWrapper,
    MountainCarPartialObservableWrapper,
    MountainCarSparseRewardWrapper,
    PendulumPartialObservableWrapper,
    PendulumSparseRewardWrapper,
    PuckWorldPartialObservableWrapper,
    PuckWorldSafetyWrapper,
    PuckWorldSparseRewardWrapper,
)
from pearl.user_envs.wrappers.dynamic_action_env import DynamicActionSpaceWrapper
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment


DQN_method = {
    "name": "DQN",
    "policy_learner": DeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
DQN_LSTM_method = {
    "name": "DQN_LSTM",
    "policy_learner": DeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {
        "hidden_dim": 128,
        "history_length": 4,
        "num_layers": 1,
    },
}
CDQN_method = {
    "name": "Conservative DQN",
    "policy_learner": DeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
        "is_conservative": True,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
DDQN_method = {
    "name": "DoubleDQN",
    "policy_learner": DoubleDQN,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
SARSA_method = {
    "name": "SARSA",
    "policy_learner": DeepSARSA,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOnPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
REINFORCE_method = {
    "name": "REINFORCE",
    "policy_learner": REINFORCE,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 1,
    },
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
REINFORCE_dynamic_method = {
    "name": "REINFORCE_dynamic",
    "policy_learner": REINFORCE,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 1,
    },
    "actor_network_type": DynamicActionActorNetwork,
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
DuelingDQN_method = {
    "name": "DuelingDQN",
    "policy_learner": DeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "network_module": DuelingQValueNetwork,
    "network_args": {"hidden_dims": [64, 64], "output_dim": 1},
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
QRDQN_method = {
    "name": "QRDQN var coeff = 0",
    "policy_learner": QuantileRegressionDeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "safety_module": QuantileNetworkMeanVarianceSafetyModule,
    "safety_module_args": {"variance_weighting_coefficient": 0.0},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
QRDQN_var_coeff_05_method = {
    "name": "QRDQN var coeff = 0.5",
    "policy_learner": QuantileRegressionDeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "safety_module": QuantileNetworkMeanVarianceSafetyModule,
    "safety_module_args": {"variance_weighting_coefficient": 0.5},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
QRDQN_var_coeff_2_method = {
    "name": "QRDQN var coeff = 2",
    "policy_learner": QuantileRegressionDeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "safety_module": QuantileNetworkMeanVarianceSafetyModule,
    "safety_module_args": {"variance_weighting_coefficient": 2.0},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
BootstrappedDQN_method = {
    "name": "BootstrappedDQN",
    "policy_learner": BootstrappedDQN,
    "policy_learner_args": {
        "training_rounds": 1,
        "batch_size": 32,
    },
    "replay_buffer": BootstrapReplayBuffer,
    "replay_buffer_args": {
        "capacity": 50000,
        "p": 1.0,
        "ensemble_size": 10,
    },
    "network_module": EnsembleQValueNetwork,
    "network_args": {
        "ensemble_size": 10,
        "output_dim": 1,
        "hidden_dims": [64, 64],
        "prior_scale": 0.0,
    },
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
BootstrappedDQN_ensemble_1_method = {
    "name": "BootstrappedDQN_ensemble_1",
    "policy_learner": BootstrappedDQN,
    "policy_learner_args": {
        "training_rounds": 1,
        "batch_size": 32,
    },
    "replay_buffer": BootstrapReplayBuffer,
    "replay_buffer_args": {
        "capacity": 50000,
        "p": 1.0,
        "ensemble_size": 1,
    },
    "network_module": EnsembleQValueNetwork,
    "network_args": {
        "ensemble_size": 1,
        "output_dim": 1,
        "hidden_dims": [64, 64],
        "prior_scale": 0.0,
    },
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
PPO_method = {
    "name": "PPO",
    "policy_learner": ProximalPolicyOptimization,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 50,
        "batch_size": 32,
        "epsilon": 0.1,
    },
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
PPO_LSTM_method = {
    "name": "PPO_LSTM",
    "policy_learner": ProximalPolicyOptimization,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 10,
        "batch_size": 32,
        "epsilon": 0.1,
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 1e-4,
        "entropy_bonus_scaling": 0.01,
    },
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {
        "hidden_dim": 128,
        "history_length": 4,
        "num_layers": 1,
    },
}
PPO_dynamic_method = {
    "name": "PPO_dynamic",
    "policy_learner": ProximalPolicyOptimization,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 50,
        "batch_size": 32,
        "epsilon": 0.1,
    },
    "actor_network_type": DynamicActionActorNetwork,
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
SAC_method = {
    "name": "SAC",
    "policy_learner": SoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
        "entropy_coef": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
SAC_LSTM_method = {
    "name": "SAC_LSTM",
    "policy_learner": SoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64, 64],
        "critic_hidden_dims": [64, 64, 64],
        "training_rounds": 1,
        "batch_size": 100,
        "entropy_coef": 0.1,
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 3e-4,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {
        "hidden_dim": 128,
        "history_length": 4,
        "num_layers": 1,
    },
}

SAC_dynamic_method = {
    "name": "SAC_dynamic",
    "policy_learner": SoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
        "entropy_coef": 0.1,
    },
    "actor_network_type": DynamicActionActorNetwork,
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
IQL_online_method = {
    "name": "IQL",
    "policy_learner": ImplicitQLearning,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "value_critic_hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
        "expectile": 0.75,
        "critic_soft_update_tau": 0.005,
        "advantage_clamp": 100.0,
        "temperature_advantage_weighted_regression": 3.0,
        "value_critic_learning_rate": 1e-3,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "actor_network_type": VanillaActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "value_network_type": VanillaValueNetwork,
        "discount_factor": 0.99,
    },
    "exploration_module": PropensityExploration,
    "exploration_module_args": {},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}

IQL_offline_method = {
    "name": "IQL",
    "policy_learner": ImplicitQLearning,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "value_critic_hidden_dims": [256, 256],
        "training_rounds": 2,
        "batch_size": 256,
        "expectile": 0.75,
        "critic_soft_update_tau": 0.05,
        "temperature_advantage_weighted_regression": 3.0,
        "value_critic_learning_rate": 1e-3,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 1e-4,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "state_value_network_type": VanillaValueNetwork,
        "discount_factor": 0.99,
    },
    "exploration_module": None,
    "exploration_module_args": {},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}

CIQL_online_method = {
    "name": "Continuous IQL",
    "policy_learner": ImplicitQLearning,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "value_critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "expectile": 0.75,
        "critic_soft_update_tau": 0.05,
        "advantage_clamp": 100.0,
        "temperature_advantage_weighted_regression": 3.0,
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 3e-4,
        "value_critic_learning_rate": 1e-3,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "value_network_type": VanillaValueNetwork,
        "discount_factor": 0.99,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}
DDPG_method = {
    "name": "DDPG",
    "policy_learner": DeepDeterministicPolicyGradient,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "actor_soft_update_tau": 0.005,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}

DDPG_LSTM_method = {
    "name": "DDPG_LSTM",
    "policy_learner": DeepDeterministicPolicyGradient,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 50,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "actor_soft_update_tau": 0.05,
        "critic_soft_update_tau": 0.05,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "discount_factor": 0.99,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {
        "hidden_dim": 128,
        "history_length": 4,
        "num_layers": 1,
    },
}

TD3_method = {
    "name": "TD3",
    "policy_learner": TD3,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "actor_soft_update_tau": 0.005,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
        "actor_update_freq": 2,
        "actor_update_noise": 0.2,
        "actor_update_noise_clip": 0.5,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}

TD3_LSTM_method = {
    "name": "TD3_LSTM",
    "policy_learner": TD3,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 50,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "actor_soft_update_tau": 0.05,
        "critic_soft_update_tau": 0.05,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "discount_factor": 0.99,
        "actor_update_freq": 2,
        "actor_update_noise": 0.2,
        "actor_update_noise_clip": 0.5,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {
        "hidden_dim": 128,
        "history_length": 4,
        "num_layers": 1,
    },
}

CSAC_method = {
    "name": "ContinuousSAC",
    "policy_learner": ContinuousSoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "entropy_autotune": False,
        "entropy_coef": 0.25,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 5e-4,
        "actor_network_type": GaussianActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "discount_factor": 0.99,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}

CSAC_LSTM_method = {
    "name": "ContinuousSACLSTM",
    "policy_learner": ContinuousSoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "entropy_autotune": False,
        "entropy_coef": 0.25,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 5e-4,
        "actor_network_type": GaussianActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "discount_factor": 0.99,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {
        "hidden_dim": 128,
        "history_length": 4,
        "num_layers": 1,
    },
}


RCDDPG_method_const_0_2 = {
    "name": "RCDDPG $\\alpha$=0.2",
    "policy_learner": DeepDeterministicPolicyGradient,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "actor_soft_update_tau": 0.005,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000, "has_cost_available": True},
    "safety_module": RCSafetyModuleCostCriticContinuousAction,
    "safety_module_args": {
        "constraint_value": 0.2,
        "lambda_constraint_ub_value": 200.0,
        "lr_lambda": 1e-3,
    },
}

RCDDPG_method_const_0_05 = deepcopy(RCDDPG_method_const_0_2)
RCDDPG_method_const_0_05["name"] = "RCDDPG $\\alpha$=0.05"
RCDDPG_method_const_0_05["safety_module_args"]["constraint_value"] = 0.05

RCDDPG_method_const_0_1 = deepcopy(RCDDPG_method_const_0_2)
RCDDPG_method_const_0_1["name"] = "RCDDPG $\\alpha$=0.1"
RCDDPG_method_const_0_1["safety_module_args"]["constraint_value"] = 0.1

RCDDPG_method_const_0_4 = deepcopy(RCDDPG_method_const_0_2)
RCDDPG_method_const_0_4["name"] = "RCDDPG $\\alpha$=0.4"
RCDDPG_method_const_0_4["safety_module_args"]["constraint_value"] = 0.4

RCDDPG_method_const_0_8 = deepcopy(RCDDPG_method_const_0_2)
RCDDPG_method_const_0_8["name"] = "RCDDPG $\\alpha$=0.8"
RCDDPG_method_const_0_8["safety_module_args"]["constraint_value"] = 0.8


RCTD3_method_const_0_2 = {
    "name": "RCTD3 $\\alpha$=0.2",
    "policy_learner": TD3,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "actor_network_type": VanillaContinuousActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "actor_soft_update_tau": 0.005,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
        "actor_update_freq": 2,
        "actor_update_noise": 0.2,
        "actor_update_noise_clip": 0.5,
    },
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000, "has_cost_available": True},
    "safety_module": RCSafetyModuleCostCriticContinuousAction,
    "safety_module_args": {
        "constraint_value": 0.2,
        "lambda_constraint_ub_value": 200.0,
        "lr_lambda": 1e-3,
    },
}

RCTD3_method_const_0_05 = deepcopy(RCTD3_method_const_0_2)
RCTD3_method_const_0_05["name"] = "RCTD3 $\\alpha$=0.05"
RCTD3_method_const_0_05["safety_module_args"]["constraint_value"] = 0.05

RCTD3_method_const_0_1 = deepcopy(RCTD3_method_const_0_2)
RCTD3_method_const_0_1["name"] = "RCTD3 $\\alpha$=0.1"
RCTD3_method_const_0_1["safety_module_args"]["constraint_value"] = 0.1

RCTD3_method_const_0_4 = deepcopy(RCTD3_method_const_0_2)
RCTD3_method_const_0_4["name"] = "RCTD3 $\\alpha$=0.4"
RCTD3_method_const_0_4["safety_module_args"]["constraint_value"] = 0.4

RCTD3_method_const_0_8 = deepcopy(RCTD3_method_const_0_2)
RCTD3_method_const_0_8["name"] = "RCTD3 $\\alpha$=0.8"
RCTD3_method_const_0_8["safety_module_args"]["constraint_value"] = 0.8

RCCSAC_method_const_0_2 = {
    "name": "RCContinuousSAC $\\alpha$=0.2",
    "policy_learner": ContinuousSoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "entropy_autotune": False,
        "entropy_coef": 0.25,
        "critic_soft_update_tau": 0.005,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 5e-4,
        "actor_network_type": GaussianActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "discount_factor": 0.99,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000, "has_cost_available": True},
    "safety_module": RCSafetyModuleCostCriticContinuousAction,
    "safety_module_args": {
        "constraint_value": 0.2,
        "lambda_constraint_ub_value": 200.0,
        "lr_lambda": 1e-3,
    },
}


RCCSAC_method_const_0_05 = deepcopy(RCCSAC_method_const_0_2)
RCCSAC_method_const_0_05["name"] = "RCCSAC $\\alpha$=0.05"
RCCSAC_method_const_0_05["safety_module_args"]["constraint_value"] = 0.05

RCCSAC_method_const_0_1 = deepcopy(RCCSAC_method_const_0_2)
RCCSAC_method_const_0_1["name"] = "RCCSAC $\\alpha$=0.1"
RCCSAC_method_const_0_1["safety_module_args"]["constraint_value"] = 0.1

RCCSAC_method_const_0_4 = deepcopy(RCCSAC_method_const_0_2)
RCCSAC_method_const_0_4["name"] = "RCCSAC $\\alpha$=0.4"
RCCSAC_method_const_0_4["safety_module_args"]["constraint_value"] = 0.4

RCCSAC_method_const_0_8 = deepcopy(RCCSAC_method_const_0_2)
RCCSAC_method_const_0_8["name"] = "RCCSAC $\\alpha$=0.8"
RCCSAC_method_const_0_8["safety_module_args"]["constraint_value"] = 0.8


RCREINFORCE_method_const_0_2 = {
    "name": "REINFORCE $\\alpha$=0.2",
    "policy_learner": REINFORCE,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 1,
    },
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000, "has_cost_available": True},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "safety_module": RCSafetyModuleCostCriticContinuousAction,
    "safety_module_args": {
        "constraint_value": 0.2,
        "lambda_constraint_ub_value": 200.0,
        "lr_lambda": 1e-3,
    },
}

RCSAC_method_const_0_2 = {
    "name": "SAC",
    "policy_learner": SoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
        "entropy_coef": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000, "has_cost_available": True},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "safety_module": RCSafetyModuleCostCriticContinuousAction,
    "safety_module_args": {
        "constraint_value": 0.2,
        "lambda_constraint_ub_value": 200.0,
        "lr_lambda": 1e-3,
    },
}

RCPPO_method_const_0_2 = {
    "name": "PPO",
    "policy_learner": ProximalPolicyOptimization,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "training_rounds": 50,
        "batch_size": 32,
        "epsilon": 0.1,
    },
    "replay_buffer": OnPolicyEpisodicReplayBuffer,
    "replay_buffer_args": {"capacity": 50000, "has_cost_available": True},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "safety_module": RCSafetyModuleCostCriticContinuousAction,
    "safety_module_args": {
        "constraint_value": 0.2,
        "lambda_constraint_ub_value": 200.0,
        "lr_lambda": 1e-3,
    },
}


all_online_discrete_control_methods = [
    DQN_method,
    CDQN_method,
    DDQN_method,
    SARSA_method,
    REINFORCE_method,
    DuelingDQN_method,
    QRDQN_method,
    BootstrappedDQN_method,
    PPO_method,
    SAC_method,
    IQL_online_method,
]
all_online_ac_discrete_control_methods = [
    REINFORCE_method,
    PPO_method,
    SAC_method,
    IQL_online_method,
]
all_online_continuous_control_methods = [
    DDPG_method,
    TD3_method,
    CSAC_method,
    CIQL_online_method,
]
all_discrete_control_envs = [
    "CartPole-v0",
    "Acrobot-v1",
    "MountainCar-v0",
]
all_continuous_control_envs = [
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "HalfCheetah-v4",
    "Ant-v4",
    "Hopper-v4",
    "Walker2d-v4",
]
all_continuous_control_w_cost_envs = [
    # "MountainCarContinuous-v0_w_cost",
    # "Pendulum-v1_w_cost",
    # "HalfCheetah-v4_w_cost",
    # "Ant-v4_w_cost",
    "Hopper-v4_w_cost",
    # "Walker2d-v4_w_cost",
]
all_dynamic_action_space_envs = [
    "Acrobot-DynamicAction-v1",
]
classic_continuous_control_envs = [
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]
mujoco_envs = [
    "HalfCheetah-v4",
    "Ant-v4",
    "Hopper-v4",
    "Walker2d-v4",
]
all_partial_observable_discrete_control_envs = [
    "CartPole-PO-v0",
    "Acrobot-PO-v1",
    "MountainCar-PO-v0",
    "PuckWorld-PLE-500-PO-v0",
]
all_partial_observable_continuous_control_envs = [
    "Pendulum-PO-v1",
]
all_sparse_reward_discrete_control_envs = [
    "Acrobot-SR-v1",
    "MountainCar-SR-v0",
    "PuckWorld-PLE-500-SR-v0",
]
all_sparse_reward_continuous_control_envs = [
    "Pendulum-SR-v1",
]
all_safety_discrete_control_envs = [
    "StochMDP-v0",
    "PuckWorld-PLE-500-SF-v0",
]

mujoco_steps = 500000
classic_control_steps = 100000
ple_steps = 500000
num_runs = 4
print_every_x_steps = 1000

# the following sets of experiments are used to generate figures and should be fixed

rctd3_walker = [
    {
        "exp_name": "rctd3_walker",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCTD3_method_const_0_05,
            RCTD3_method_const_0_1,
            RCTD3_method_const_0_2,
            RCTD3_method_const_0_4,
            RCTD3_method_const_0_8,
            TD3_method,
        ],
        "device_id": 3,
    }
    for env_name in [
        "Walker2d-v4_w_cost",
    ]
]

rcddpg_walker = [
    {
        "exp_name": "rcddpg_walker",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCDDPG_method_const_0_05,
            RCDDPG_method_const_0_1,
            RCDDPG_method_const_0_2,
            RCDDPG_method_const_0_4,
            RCDDPG_method_const_0_8,
            DDPG_method,
        ],
        "device_id": 3,
    }
    for env_name in [
        "Walker2d-v4_w_cost",
    ]
]

rccsac_walker = [
    {
        "exp_name": "rccsac_walker",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCCSAC_method_const_0_05,
            RCCSAC_method_const_0_1,
            RCCSAC_method_const_0_2,
            RCCSAC_method_const_0_4,
            RCCSAC_method_const_0_8,
            CSAC_method,
        ],
        "device_id": 3,
    }
    for env_name in [
        "Walker2d-v4_w_cost",
    ]
]


rcddpg_hopper = [
    {
        "exp_name": "rcddpg_hopper",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCDDPG_method_const_0_05,
            RCDDPG_method_const_0_1,
            RCDDPG_method_const_0_2,
            RCDDPG_method_const_0_4,
            RCDDPG_method_const_0_8,
            DDPG_method,
        ],
        "device_id": 2,
    }
    for env_name in [
        "Hopper-v4_w_cost",
    ]
]

rctd3_hopper = [
    {
        "exp_name": "rctd3_hopper_part_1",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCTD3_method_const_0_05,
            RCTD3_method_const_0_1,
            RCTD3_method_const_0_2,
            RCTD3_method_const_0_4,
            RCTD3_method_const_0_8,
            TD3_method,
        ],
        "device_id": 2,
    }
    for env_name in [
        "Hopper-v4_w_cost",
    ]
]

rccsac_hopper = [
    {
        "exp_name": "rccsac_hopper_part_1",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCCSAC_method_const_0_05,
            RCCSAC_method_const_0_1,
            RCCSAC_method_const_0_2,
            RCCSAC_method_const_0_4,
            RCCSAC_method_const_0_8,
            CSAC_method,
        ],
        "device_id": 2,
    }
    for env_name in [
        "Hopper-v4_w_cost",
    ]
]


rctd3_ant = [
    {
        "exp_name": "rctd3_ant",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCTD3_method_const_0_05,
            RCTD3_method_const_0_1,
            RCTD3_method_const_0_2,
            RCTD3_method_const_0_4,
            RCTD3_method_const_0_8,
            TD3_method,
        ],
        "device_id": 0,
    }
    for env_name in [
        "Ant-v4_w_cost",
    ]
]

rcddpg_ant = [
    {
        "exp_name": "rcddpg_ant_part_1",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCDDPG_method_const_0_05,
            RCDDPG_method_const_0_1,
            RCDDPG_method_const_0_2,
            RCDDPG_method_const_0_4,
            RCDDPG_method_const_0_8,
            DDPG_method,
        ],
        "device_id": 0,
    }
    for env_name in [
        "Ant-v4_w_cost",
    ]
]

rccsac_ant = [
    {
        "exp_name": "rccsac_ant",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCCSAC_method_const_0_05,
            RCCSAC_method_const_0_1,
            RCCSAC_method_const_0_2,
            RCCSAC_method_const_0_4,
            RCCSAC_method_const_0_8,
            CSAC_method,
        ],
        "device_id": 0,
    }
    for env_name in [
        "Ant-v4_w_cost",
    ]
]

rctd3_half_cheetah = [
    {
        "exp_name": "rctd3_half_cheetah",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCTD3_method_const_0_05,
            RCTD3_method_const_0_1,
            RCTD3_method_const_0_2,
            RCTD3_method_const_0_4,
            RCTD3_method_const_0_8,
            TD3_method,
        ],
        "device_id": 1,
    }
    for env_name in [
        "HalfCheetah-v4_w_cost",
    ]
]


rcddpg_half_cheetah = [
    {
        "exp_name": "rcddpg_half_cheetah",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCDDPG_method_const_0_05,
            RCDDPG_method_const_0_1,
            RCDDPG_method_const_0_2,
            RCDDPG_method_const_0_4,
            RCDDPG_method_const_0_8,
            DDPG_method,
        ],
        "device_id": 1,
    }
    for env_name in [
        "HalfCheetah-v4_w_cost",
    ]
]

rccsac_half_cheetah = [
    {
        "exp_name": "rccsac_half_cheetah",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            RCCSAC_method_const_0_05,
            RCCSAC_method_const_0_1,
            RCCSAC_method_const_0_2,
            RCCSAC_method_const_0_4,
            RCCSAC_method_const_0_8,
            CSAC_method,
        ],
        "device_id": 1,
    }
    for env_name in [
        "HalfCheetah-v4_w_cost",
    ]
]

benchmark_cartpole_v1_part_1 = [
    {
        "exp_name": "benchmark_cartpole_v1_part_1",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": classic_control_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            DQN_method,
            CDQN_method,
            DDQN_method,
            SARSA_method,
            REINFORCE_method,
            DuelingDQN_method,
        ],
        "device_id": 0,
    }
    for env_name in [
        "CartPole-v1",
    ]
]

benchmark_cartpole_v1_part_2 = [
    {
        "exp_name": "benchmark_cartpole_v1_part_2",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": classic_control_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            QRDQN_method,
            BootstrappedDQN_method,
            PPO_method,
            SAC_method,
            IQL_online_method,
        ],
        "device_id": 1,
    }
    for env_name in [
        "CartPole-v1",
    ]
]

benchmark_acrobot_v1_part_1 = [
    {
        "exp_name": "benchmark_acrobot_v1_part_1",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": classic_control_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            DQN_method,
            CDQN_method,
            DDQN_method,
            SARSA_method,
            REINFORCE_method,
            DuelingDQN_method,
        ],
        "device_id": 0,
    }
    for env_name in [
        "Acrobot-v1",
    ]
]

benchmark_acrobot_v1_part_2 = [
    {
        "exp_name": "benchmark_acrobot_v1_part_2",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": classic_control_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            QRDQN_method,
            BootstrappedDQN_method,
            PPO_method,
            SAC_method,
            IQL_online_method,
        ],
        "device_id": 1,
    }
    for env_name in [
        "Acrobot-v1",
    ]
]

benchmark_halfcheetah_v4 = [
    {
        "exp_name": "benchmark_halfcheetah_v4",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": all_online_continuous_control_methods,
        "device_id": 0,
    }
    for env_name in [
        "HalfCheetah-v4",
    ]
]

benchmark_pendulum_v1_lstm = [
    {
        "exp_name": "benchmark_pendulum_v1_lstm",
        "env_name": env_name,
        "num_runs": 4,
        "num_steps": 100000,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            DDPG_LSTM_method,
            CSAC_LSTM_method,
            TD3_LSTM_method,
        ],
        "device_id": 2,
    }
    for env_name in [
        "Pendulum-v1",
    ]
]
benchmark_ant_v4 = [
    {
        "exp_name": "benchmark_ant_v4",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": all_online_continuous_control_methods,
        "device_id": 1,
    }
    for env_name in [
        "Ant-v4",
    ]
]

benchmark_hopper_v4 = [
    {
        "exp_name": "benchmark_hopper_v4",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": all_online_continuous_control_methods,
        "device_id": 0,
    }
    for env_name in [
        "Hopper-v4",
    ]
]

benchmark_walker2d_v4 = [
    {
        "exp_name": "benchmark_walker2d_v4",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": mujoco_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": all_online_continuous_control_methods,
        "device_id": 1,
    }
    for env_name in [
        "Walker2d-v4",
    ]
]

# the following experiments are not used to generate figures and may change over time

test_qrdqn = [
    {
        "exp_name": "test_qrdqn",
        "env_name": env_name,
        "num_runs": num_runs,
        "num_steps": classic_control_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            QRDQN_method,
            QRDQN_var_coeff_05_method,
            QRDQN_var_coeff_2_method,
        ],
    }
    for env_name in [
        "PuckWorld-PLE-500-SF-v0",
        "MeanVarBandit-v0",
    ]
]

test_dynamic_action_space = [
    {
        "exp_name": "test_dynamic_action",
        "env_name": env_name,
        "num_runs": 3,
        "num_steps": classic_control_steps,
        "print_every_x_steps": print_every_x_steps,
        "record_period": 1000,
        "methods": [
            DQN_method,
            REINFORCE_dynamic_method,
            PPO_dynamic_method,
            SAC_dynamic_method,
        ],
        "device_id": 0,
    }
    for env_name in [
        "Acrobot-DynamicAction-v1",
        "CartPole-DynamicAction-v1",
    ]
]


def get_env(env_name: str) -> GymEnvironment:
    """
    attach a versatility wrapper to the environment
    """
    if env_name == "CartPole-PO-v0":
        return GymEnvironment(
            CartPolePartialObservableWrapper(
                gym.make("CartPole-v0"), time_between_two_valid_obs=2
            )
        )
    elif env_name == "Pendulum-PO-v1":
        return GymEnvironment(
            PendulumPartialObservableWrapper(
                gym.make("Pendulum-v1"), time_between_two_valid_obs=2
            )
        )
    elif env_name == "Acrobot-PO-v1":
        return GymEnvironment(
            AcrobotPartialObservableWrapper(
                gym.make("Acrobot-v1"), time_between_two_valid_obs=2
            )
        )
    elif env_name == "MountainCar-PO-v0":
        return GymEnvironment(
            MountainCarPartialObservableWrapper(
                gym.make("MountainCar-v0"), time_between_two_valid_obs=2
            )
        )
    elif env_name == "Acrobot-SR-v1":
        return GymEnvironment(AcrobotSparseRewardWrapper(gym.make("Acrobot-v1")))
    elif env_name == "Pendulum-SR-v1":
        return GymEnvironment(PendulumSparseRewardWrapper(gym.make("Pendulum-v1")))
    elif env_name == "MountainCar-SR-v0":
        return GymEnvironment(
            MountainCarSparseRewardWrapper(gym.make("MountainCar-v0"))
        )
    elif env_name == "PuckWorld-PLE-500-SR-v0":
        return GymEnvironment(
            PuckWorldSparseRewardWrapper(gym.make("PuckWorld-PLE-500-v0"))
        )
    elif env_name == "PuckWorld-PLE-500-SF-v0":
        return GymEnvironment(PuckWorldSafetyWrapper(gym.make("PuckWorld-PLE-500-v0")))
    elif env_name == "PuckWorld-PLE-500-PO-v0":
        return GymEnvironment(
            PuckWorldPartialObservableWrapper(gym.make("PuckWorld-PLE-500-v0"))
        )
    elif env_name == "Acrobot-DynamicAction-v1":
        return GymEnvironment(DynamicActionSpaceWrapper(gym.make("Acrobot-v1")))
    elif env_name == "CartPole-DynamicAction-v1":
        return GymEnvironment(DynamicActionSpaceWrapper(gym.make("CartPole-v1")))
    elif env_name[-7:] == "_w_cost":
        env_name = env_name[:-7]
        return GymEnvironment(GymAvgTorqueWrapper(gym.make(env_name)))
    else:
        return GymEnvironment(env_name)
