#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
To RUN: (assume under fbcode/)
    first time:
    buck2 run pearl/scripts:figure_gen <num_episodes>
    second time, something like:
    ../buck-out/v2/gen/fbcode/e97a0788aa35bdc8/pearl/scripts/__figure_gen__/figure_gen.par
"""
import logging
import sys

import matplotlib.pyplot as plt
from pearl.gym.gym_environment import GymEnvironment
from pearl.online_learning.online_learning import online_learning_returns
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.deep_q_learning import DeepQLearning
from pearl.replay_buffer.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer


def main():
    logging.basicConfig(level=logging.DEBUG)
    env = GymEnvironment("CartPole-v1")

    assert len(sys.argv) <= 2, "Usage: <figure_gen> <num_episodes>"
    num_episodes = 1000  # default value
    if len(sys.argv) == 2:
        num_episodes = int(sys.argv[1])

    agent = PearlAgent(
        policy_learner=DeepQLearning(
            env.observation_space.shape[0],
            env.action_space,
            [64, 64],
            training_rounds=20,
        ),
        replay_buffer=FIFOOffPolicyReplayBuffer(10000),
    )
    returns = online_learning_returns(
        agent,
        env,
        number_of_episodes=num_episodes,
        learn_after_episode=True,
    )
    plt.plot(returns, label="vanilla dqn")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    agent = PearlAgent(
        policy_learner=DeepQLearning(
            env.observation_space.shape[0],
            env.action_space,
            [64, 64],
            training_rounds=20,
            double=True,
        ),
        replay_buffer=FIFOOffPolicyReplayBuffer(10000),
    )
    returns = online_learning_returns(
        agent,
        env,
        number_of_episodes=num_episodes,
        learn_after_episode=True,
    )
    plt.plot(returns, label="double dqn")
    plt.savefig("returns2.png")


if __name__ == "__main__":
    main()
