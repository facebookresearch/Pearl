# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from typing import Any, Dict, Tuple

import torch
from pearl.action_representation_modules.binary_action_representation_module import (
    BinaryActionTensorRepresentationModule,
)
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import (
    NeuralLinearBandit,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import (
    FastCBExploration,
    SquareCBExploration,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (
    ThompsonSamplingExplorationLinear,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
)
from pearl.utils.instantiations.environments.contextual_bandit_uci_environment import (
    SLCBEnvironment,
)

DATA_PATH: str = "./utils/instantiations/environments/uci_datasets"

"""
Experiment config
"""
run_config: Dict[str, Any] = {
    "T": 10000,
    "training_rounds": 10,
    "num_of_experiments": 5,
}
"""
UCI Dataset configs
"""
# satimage uci dataset
satimage_uci: str = os.path.join(DATA_PATH, "satimage/sat.trn")
satimage_uci_dict: Dict[str, Any] = {
    "path_filename": satimage_uci,
    "action_embeddings": "discrete",
    "delim_whitespace": True,
    "ind_to_drop": [],
    "target_column": 36,
}

# letter uci dataset
letter_uci: str = os.path.join(DATA_PATH, "letter/letter-recognition.data")
letter_uci_dict: Dict[str, Any] = {
    "path_filename": letter_uci,
    "action_embeddings": "discrete",
}

# yeast uci dataset
yeast_uci: str = os.path.join(DATA_PATH, "yeast/yeast.data")
yeast_uci_dict: Dict[str, Any] = {
    "path_filename": yeast_uci,
    "action_embeddings": "discrete",
    "delim_whitespace": True,
    "ind_to_drop": [0],
    "target_column": 8,
}

# pendigits uci dataset
pendigits_uci: str = os.path.join(DATA_PATH, "pendigits/pendigits.tra")
pendigits_uci_dict: Dict[str, Any] = {
    "path_filename": pendigits_uci,
    "action_embeddings": "discrete",
    "delim_whitespace": False,
    "ind_to_drop": [],
    "target_column": 16,
}

"""
CB config
"""


def return_neural_squarecb_config(
    env: SLCBEnvironment,
    run_config: Dict[str, Any] = run_config,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    feature_dim: int = env.observation_dim
    dim_actions: int = env.bits_num

    policy_learner_dict: Dict[str, Any] = {
        "name": "NeuralBandit",
        "method": NeuralBandit,
        "params": {
            "feature_dim": feature_dim + dim_actions,
            "hidden_dims": [64, 16],
            "learning_rate": 0.01,
            "batch_size": 128,
            "training_rounds": run_config["training_rounds"],
            "action_representation_module": BinaryActionTensorRepresentationModule(
                bits_num=dim_actions
            ),
        },
    }

    gamma: float = (
        torch.sqrt(torch.tensor(run_config["T"]) * (feature_dim + dim_actions)).item()
        * 10.0
    )
    exploration_module_dict: Dict[str, Any] = {
        "name": "SquareCBExploration",
        "method": SquareCBExploration,
        "params": {"gamma": gamma},
    }

    return policy_learner_dict, exploration_module_dict


def return_neural_lin_ucb_config(
    env: SLCBEnvironment,
    run_config: Dict[str, Any] = run_config,
    alpha: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    feature_dim: int = env.observation_dim
    dim_actions: int = env.bits_num

    policy_learner_dict: Dict[str, Any] = {
        "name": "NeuralLinearBandit",
        "method": NeuralLinearBandit,
        "params": {
            "feature_dim": feature_dim + dim_actions,
            "hidden_dims": [64, 16],
            "learning_rate": 0.01,
            "batch_size": 128,
            "training_rounds": run_config["training_rounds"],
            "action_representation_module": BinaryActionTensorRepresentationModule(
                bits_num=dim_actions
            ),
        },
    }

    exploration_module_dict: Dict[str, Any] = {
        "name": "UCBExploration",
        "method": UCBExploration,
        "params": {"alpha": alpha},
    }

    return policy_learner_dict, exploration_module_dict


def return_neural_lin_ts_config(
    env: SLCBEnvironment,
    run_config: Dict[str, Any] = run_config,
    alpha: float = 0.25,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    feature_dim: int = env.observation_dim
    dim_actions: int = env.bits_num

    policy_learner_dict: Dict[str, Any] = {
        "name": "NeuralTSBandit",
        "method": NeuralLinearBandit,
        "params": {
            "feature_dim": feature_dim + dim_actions,
            "hidden_dims": [64, 16],
            "learning_rate": 0.01,
            "batch_size": 128,
            "training_rounds": run_config["training_rounds"],
            "action_representation_module": BinaryActionTensorRepresentationModule(
                bits_num=dim_actions
            ),
        },
    }

    exploration_module_dict: Dict[str, Any] = {
        "name": "ThompsonSamplingExplorationLinear",
        "method": ThompsonSamplingExplorationLinear,
        "params": {},
    }

    return policy_learner_dict, exploration_module_dict


def return_offline_eval_config(
    env: SLCBEnvironment,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dim_actions: int = env.bits_num
    exploration_module_dict = {"name": "None"}
    policy_learner_dict = {
        "name": "Offline",
        "params": {
            "hidden_dim": [64, 16],
            "num_eval_steps": 5000,
            "action_representation_module": BinaryActionTensorRepresentationModule(
                bits_num=dim_actions
            ),
        },
    }

    return policy_learner_dict, exploration_module_dict


def return_neural_fastcb_config(
    env: SLCBEnvironment,
    run_config: Dict[str, Any] = run_config,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    feature_dim: int = env.observation_dim
    dim_actions: int = env.bits_num

    policy_learner_dict: Dict[str, Any] = {
        "name": "NeuralBandit",
        "method": NeuralBandit,
        "params": {
            "feature_dim": feature_dim + dim_actions,
            "hidden_dims": [64, 16],
            "learning_rate": 0.01,
            "batch_size": 128,
            "training_rounds": run_config["training_rounds"],
            "action_representation_module": BinaryActionTensorRepresentationModule(
                bits_num=dim_actions
            ),
        },
    }

    gamma: float = (
        torch.sqrt(torch.tensor(run_config["T"]) * (feature_dim + dim_actions)).item()
        * 10.0
    )
    exploration_module_dict: Dict[str, Any] = {
        "name": "FastCBExploration",
        "method": FastCBExploration,
        "params": {"gamma": gamma},
    }

    return policy_learner_dict, exploration_module_dict
