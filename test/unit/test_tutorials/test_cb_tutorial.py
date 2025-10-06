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
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (  # noqa E501
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
