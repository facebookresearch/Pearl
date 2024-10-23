# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from .actor_critic_base import ActorCriticBase
from .bootstrapped_dqn import BootstrappedDQN
from .ddpg import DeepDeterministicPolicyGradient
from .deep_q_learning import DeepQLearning
from .deep_sarsa import DeepSARSA
from .deep_td_learning import DeepTDLearning
from .double_dqn import DoubleDQN
from .implicit_q_learning import ImplicitQLearning
from .ppo import ProximalPolicyOptimization
from .quantile_regression_deep_q_learning import QuantileRegressionDeepQLearning
from .quantile_regression_deep_td_learning import QuantileRegressionDeepTDLearning
from .reinforce import REINFORCE
from .soft_actor_critic import SoftActorCritic
from .soft_actor_critic_continuous import ContinuousSoftActorCritic
from .tabular_q_learning import TabularQLearning
from .td3 import TD3


__all__ = [
    "ActorCriticBase",
    "BootstrappedDQN",
    "DeepDeterministicPolicyGradient",
    "DeepQLearning",
    "DeepSARSA",
    "DeepTDLearning",
    "DoubleDQN",
    "ImplicitQLearning",
    "ProximalPolicyOptimization",
    "PPOReplayBuffer",
    "QuantileRegressionDeepQLearning",
    "QuantileRegressionDeepTDLearning",
    "REINFORCE",
    "ContinuousSoftActorCritic",
    "SoftActorCritic",
    "TabularQLearning",
    "TD3",
]
