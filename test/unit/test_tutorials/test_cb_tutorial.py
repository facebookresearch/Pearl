# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import os
import unittest

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import (
    NeuralLinearBandit,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import (
    SquareCBExploration,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (
    ThompsonSamplingExplorationLinear,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.contextual_bandit_uci_environment import (
    SLCBEnvironment,
)
from pearl.utils.uci_data import download_uci_data

set_seed(0)


"""
This is a unit test version of the CB tutorial.
It is meant to check whether code changes break the tutorial.
It is therefore important that the tutorial and the code here are kept in sync.
As part of that synchronization, the markdown cells in the tutorial are
kept here as multi-line strings.

For it to run quickly, the number of steps used for training is reduced.
"""


class TestCBTutorials(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_cb_tutorials(self) -> None:
        # load environment
        device_id = 0 if torch.cuda.is_available() else -1

        # Download UCI dataset if doesn't exist
        uci_data_path = "./utils/instantiations/environments/uci_datasets"
        if not os.path.exists(uci_data_path):
            os.makedirs(uci_data_path)
        download_uci_data(data_path=uci_data_path)

        # Built CB environment using the pendigits UCI dataset
        pendigits_uci_dict = {
            "path_filename": os.path.join(uci_data_path, "pendigits/pendigits.tra"),
            "action_embeddings": "discrete",
            "delim_whitespace": False,
            "ind_to_drop": [],
            "target_column": 16,
        }
        env = SLCBEnvironment(**pendigits_uci_dict)  # pyre-ignore

        # experiment code
        number_of_steps = 10
        record_period = 10

        """
        SquareCB
        """
        # Create a Neural SquareCB pearl agent with 1-hot action representation
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=env.unique_labels_num,
        )

        agent = PearlAgent(
            policy_learner=NeuralBandit(
                feature_dim=env.observation_dim + env.unique_labels_num,
                hidden_dims=[2],
                training_rounds=2,
                learning_rate=0.01,
                action_representation_module=action_representation_module,
                exploration_module=SquareCBExploration(
                    gamma=env.observation_dim * env.unique_labels_num * number_of_steps
                ),
            ),
            replay_buffer=BasicReplayBuffer(100_000),
            device_id=device_id,
        )

        _ = online_learning(
            agent=agent,
            env=env,
            number_of_steps=number_of_steps,
            print_every_x_steps=10,
            record_period=record_period,
            learn_after_episode=True,
        )

        # Neural LinUCB
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=env.unique_labels_num,
        )

        agent = PearlAgent(
            policy_learner=NeuralLinearBandit(
                feature_dim=env.observation_dim + env.unique_labels_num,
                hidden_dims=[2],
                state_features_only=False,
                training_rounds=2,
                learning_rate=0.01,
                action_representation_module=action_representation_module,
                exploration_module=UCBExploration(alpha=1.0),
            ),
            replay_buffer=BasicReplayBuffer(100_000),
            device_id=device_id,
        )

        _ = online_learning(
            agent=agent,
            env=env,
            number_of_steps=number_of_steps,
            print_every_x_steps=100,
            record_period=record_period,
            learn_after_episode=True,
        )

        # Neural LinTS

        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=env.unique_labels_num,
        )

        agent = PearlAgent(
            policy_learner=NeuralLinearBandit(
                feature_dim=env.observation_dim + env.unique_labels_num,
                hidden_dims=[2],
                state_features_only=False,
                training_rounds=2,
                learning_rate=0.01,
                action_representation_module=action_representation_module,
                exploration_module=ThompsonSamplingExplorationLinear(),
            ),
            replay_buffer=BasicReplayBuffer(100_000),
            device_id=device_id,
        )

        _ = online_learning(
            agent=agent,
            env=env,
            number_of_steps=number_of_steps,
            print_every_x_steps=10,
            record_period=record_period,
            learn_after_episode=True,
        )
