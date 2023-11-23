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
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

DQN_method = {
    "name": "DQN",
    "policy_learner": DeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "agent_args": {"device_id": 0},
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}

DQN_LSTM_method = {
    "name": "LSTMDQN",
    "policy_learner": DeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "agent_args": {"device_id": 0},
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
    "history_summarization_module": LSTMHistorySummarizationModule,
    "history_summarization_module_args": {"hidden_dim": 8},
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
    "agent_args": {"device_id": 0},
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
    "agent_args": {"device_id": 0},
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
    "agent_args": {"device_id": 0},
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
    "agent_args": {"device_id": 0},
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
    "agent_args": {"device_id": 0},
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
    "name": "QRDQN",
    "policy_learner": QuantileRegressionDeepQLearning,
    "policy_learner_args": {
        "hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
    },
    "agent_args": {"device_id": 1},
    "exploration_module": EGreedyExploration,
    "exploration_module_args": {"epsilon": 0.1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "safety_module": QuantileNetworkMeanVarianceSafetyModule,
    "safety_module_args": {"variance_weighting_coefficient": 0.0},
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
    "agent_args": {"device_id": 1},
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
    "agent_args": {"device_id": 0},
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
    "agent_args": {"device_id": 1},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
}
IQL_method = {
    "name": "IQL",
    "policy_learner": ImplicitQLearning,
    "policy_learner_args": {
        "actor_hidden_dims": [64, 64],
        "critic_hidden_dims": [64, 64],
        "state_value_critic_hidden_dims": [64, 64],
        "training_rounds": 1,
        "batch_size": 32,
        "expectile": 0.7,
        "critic_soft_update_tau": 0.005,
        "advantage_clamp": 100.0,
        "temperature_advantage_weighted_regression": 3.0,
        "state_value_learning_rate": 1e-3,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "actor_network_type": VanillaActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "state_value_network_type": VanillaValueNetwork,
        "discount_factor": 0.99,
    },
    "agent_args": {"device_id": 1},
    "exploration_module": PropensityExploration,
    "exploration_module_args": {},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 50000},
    "action_representation_module": OneHotActionTensorRepresentationModule,
    "action_representation_module_args": {},
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
        "critic_soft_update_tau": 0.05,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
    },
    "agent_args": {"device_id": 0},
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
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
        "critic_soft_update_tau": 0.05,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
        "actor_update_freq": 2,
        "actor_update_noise": 0.2,
        "actor_update_noise_clip": 0.5,
    },
    "agent_args": {"device_id": 1},
    "exploration_module": NormalDistributionExploration,
    "exploration_module_args": {
        "mean": 0,
        "std_dev": 0.1,
    },
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}
CSAC_method = {
    "name": "ContinuousSAC",
    "policy_learner": ContinuousSoftActorCritic,
    "policy_learner_args": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "training_rounds": 1,
        "batch_size": 256,
        "entropy_autotune": True,
        "entropy_coef": 0.05,
        "critic_soft_update_tau": 0.05,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 1e-3,
        "actor_network_type": GaussianActorNetwork,
        "critic_network_type": VanillaQValueNetwork,
        "discount_factor": 0.99,
    },
    "agent_args": {"device_id": 0},
    "replay_buffer": FIFOOffPolicyReplayBuffer,
    "replay_buffer_args": {"capacity": 100000},
}
all_discrete_control_methods = [
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
]
all_ac_discrete_control_methods = [
    REINFORCE_method,
    PPO_method,
    SAC_method,
    IQL_method,
]
all_continuous_control_methods = [
    DDPG_method,
    TD3_method,
    CSAC_method,
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
    # "Acrobot-PO-v1",
    # "MountainCar-PO-v0",
    # "PuckWorld-PLE-500-PO-v0",
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

mujoco_steps = 2000000
classic_control_steps = 100000
ple_steps = 2000000
num_runs = 5
print_every_x_steps = 1000


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
    elif env_name[-7:] == "_w_cost":
        env_name = env_name[:-7]
        return GymEnvironment(GymAvgTorqueWrapper(gym.make(env_name)))
    else:
        return GymEnvironment(env_name)
