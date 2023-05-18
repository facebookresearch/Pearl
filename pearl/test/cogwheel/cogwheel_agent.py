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
from pearl.online_learning.online_learning import (
    episode_return,
    online_learning_returns,
)
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.deep_q_learning import DeepQLearning
from pearl.replay_buffer.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from windtunnel.cogwheel.test import cogwheel_test, CogwheelTest


class TestDeepQLearning(CogwheelTest):
    def setUp(self) -> None:
        self.env = GymEnvironment("CartPole-v1")
        self.agent = PearlAgent(
            policy_learner=DeepQLearning(
                self.env.observation_space.shape[0],
                self.env.action_space,
                [64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10000),
        )

    @cogwheel_test
    def test_deep_td_learning_online_rl(self) -> None:
        online_learning_returns(
            self.agent,
            self.env,
            number_of_episodes=1000,
            learn_after_episode=True,
        )
        learnt = 500 in [episode_return(self.agent, self.env) for _ in range(100)]
        self.assertEqual(learnt, True)
        # Should keep cartpole upright at least once in 100 episodes.


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    TestDeepQLearning().main()
