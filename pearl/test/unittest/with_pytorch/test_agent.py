#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

from pearl.gym.gym_environment import GymEnvironment
from pearl.online_learning.online_learning import online_learning_to_png_graph
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.deep_q_learning import DeepQLearning
from pearl.replay_buffer.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.test.utils import create_random_batch


class TestAgentWithPyTorch(unittest.TestCase):
    """
    A collection of Agent tests using PyTorch (this saves around 100 secs in test loading).
    For tests not involving PyTorch, use see test/without_pytorch.
    """

    def test_deep_td_learning_online_rl_sanity_check(self) -> None:
        # make sure E2E is fine
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        online_learning_to_png_graph(
            agent, env, number_of_episodes=10, learn_after_episode=True
        )

    def test_learn_batch(self) -> None:
        # simple test to ensure e2e run
        state_dim = 3
        action_dim = 3
        batch_size = 10
        batch, action_space = create_random_batch(action_dim, state_dim, batch_size)
        self.agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim,
                action_space,
                [64, 64],
            ),
        )
        self.agent.learn_batch(batch)
