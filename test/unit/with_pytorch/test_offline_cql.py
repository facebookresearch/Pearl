# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)

from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.functional_utils.experimentation.create_offline_data import (
    create_offline_data,
)
from pearl.utils.functional_utils.train_and_eval.offline_learning_and_evaluation import (
    offline_evaluation,
    offline_learning,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestOfflineCQL(unittest.TestCase):
    """
    End to end test for the offline learning pipeline. It tests the following components:
    1) Creating offline data.
    2) Loading offline data into a replay buffer.
    3) Training an offline agent (in this case CQL).
    4) Evaluating the offline agent using environment interactions.
    """

    def test_create_offline_data_and_learn_cql(self) -> None:
        # Cartpole environment
        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        max_number_actions = env.action_space.n

        # One hot representations for discrete actions
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=max_number_actions
        )

        # Data collection agent that collects and stores data from environment interactions.
        onlineDQN_agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                exploration_module=EGreedyExploration(0.5),
                training_rounds=20,
                action_representation_module=action_representation_module,
            ),
            replay_buffer=BasicReplayBuffer(1000000),
        )

        max_len_offline_data = 500

        # Collect offline data using the data collection agent
        create_offline_data(
            agent=onlineDQN_agent,
            env=env,
            save_path="",
            file_name="offline_raw_transitions_dict.pt",
            max_len_offline_data=max_len_offline_data,
            learn=True,
            learn_after_episode=True,
            evaluation_episodes=50,
        )

        # Load offline data into a Pearl replay buffer
        offline_data_replay_buffer = BasicReplayBuffer(max_len_offline_data)
        raw_transitions_buffer = torch.load(
            "offline_raw_transitions_dict.pt", weights_only=False
        )
        for transition in raw_transitions_buffer:
            offline_data_replay_buffer.push(
                state=transition["observation"],
                action=transition["action"],
                reward=transition["reward"],
                next_state=transition["next_observation"],
                curr_available_actions=transition[
                    "curr_available_actions"
                ],  # curr_available_actions
                next_available_actions=transition[
                    "next_available_actions"
                ],  # next_available_actions
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                max_number_actions=max_number_actions,
            )

        # An offline RL agent (in this case, CQL)
        conservative_agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=100,
                is_conservative=True,
                conservative_alpha=8.0,
                batch_size=128,
                action_representation_module=action_representation_module,
            ),
            replay_buffer=offline_data_replay_buffer,
        )

        # Run training for the CQL based agent using offline data
        offline_learning(
            offline_agent=conservative_agent,
            data_buffer=offline_data_replay_buffer,
            number_of_batches=10,
            seed=100,
        )

        # Run evaluation for trained agent with environment interactions
        # Note: Pearl does not have off policy evaluation (OPE) algorithms implemented yet.
        offline_evaluation(
            offline_agent=conservative_agent,
            env=env,
            number_of_episodes=10,
        )
