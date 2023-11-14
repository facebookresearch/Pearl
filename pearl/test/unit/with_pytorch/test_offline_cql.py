#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import torch
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)

from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.functional_utils.experimentation.create_offline_data import (
    create_offline_data,
)
from pearl.utils.functional_utils.train_and_eval.offline_learning_and_evaluation import (
    offline_evaluation,
    offline_learning,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment


class TestOfflineCQL(unittest.TestCase):

    # test to create and save offline data
    def test_create_offline_data_and_learn_cql(self) -> None:
        env = GymEnvironment("CartPole-v1")
        onlineDQN_agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                exploration_module=EGreedyExploration(0.5),
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(1000000),
        )

        max_len_offline_data = 500
        create_offline_data(
            agent=onlineDQN_agent,
            env=env,
            max_len_offline_data=max_len_offline_data,
        )

        offline_data_replay_buffer = FIFOOffPolicyReplayBuffer(max_len_offline_data)
        raw_transitions_buffer = torch.load("offline_raw_transitions_dict.pt")
        for transition in raw_transitions_buffer:
            offline_data_replay_buffer.push(
                transition["observation"],
                transition["action"],
                transition["reward"],
                transition["next_observation"],
                transition["curr_available_actions"],  # curr_available_actions
                transition["next_available_actions"],  # next_available_actions
                transition["action_space"],  # action_space
                transition["done"],
            )

        conservative_agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=100,
                is_conservative=True,
                conservative_alpha=8.0,
                batch_size=128,
            ),
            replay_buffer=offline_data_replay_buffer,
        )

        offline_learning(
            url="",
            offline_agent=conservative_agent,
            data_buffer=offline_data_replay_buffer,
            training_epochs=10,
        )

        offline_evaluation(
            offline_agent=conservative_agent,
            env=env,
            number_of_episodes=10,
        )
