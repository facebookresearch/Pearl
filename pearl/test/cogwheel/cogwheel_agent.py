#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Cogwheel wiki
https://www.internalfb.com/intern/wiki/Cogwheel/

To RUN:
    buck2 run //pearl:cogwheel_pearl_agent-launcher

This takes a long time (usually 3 hours on a devserver), which is why it is on cogwheel where it will not time out.
"""
import logging

from pearl.gym.gym_environment import GymEnvironment
from pearl.online_learning.online_learning import episode_return
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.deep_q_learning import DeepQLearning
from pearl.policy_learners.policy_gradient import PolicyGradient
from pearl.replay_buffer.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.replay_buffer.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
)
from windtunnel.cogwheel.test import cogwheel_test, CogwheelTest


class TestAgent(CogwheelTest):
    @cogwheel_test
    def test_dqn(self) -> None:
        """
        This test is checking if DQN will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )
        counter = 0
        while (
            episode_return(agent=agent, env=env, learn=True, learn_after_episode=True)
            != 500
        ):
            counter += 1
            # we should be able to get to 500 within 100 episodes
            # according to testplan in D46043013
            self.assertGreater(100, counter)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    TestAgent().main()
