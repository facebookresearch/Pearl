#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning

from pearl.utils.instantiations.environments.gym_environment import GymEnvironment


class TestGPUUsage(unittest.TestCase):
    """
    Test the GPU usage in pearl.
    Make sure the GPU is correctly used during model training.
    """

    def test_td_based_gpu_usage(self) -> None:
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=1,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )

        online_learning(agent, env, number_of_episodes=10)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
            self.assertTrue(allocated_memory > 0.001)
        else:
            self.assertTrue(True)

    def test_pg_based_gpu_usage(self) -> None:
        env = GymEnvironment("CartPole-v1")

        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
                batch_size=500,
                epsilon=0.1,
            ),
            replay_buffer=OnPolicyEpisodicReplayBuffer(10000),
        )

        online_learning(agent, env, number_of_episodes=10, learn_after_episode=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
            self.assertTrue(allocated_memory > 0.001)
        else:
            self.assertTrue(True)
