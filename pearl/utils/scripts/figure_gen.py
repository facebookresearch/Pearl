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
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.deep_sarsa import DeepSARSA
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.fifo_on_policy_replay_buffer import (
    FIFOOnPolicyReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

MA_WINDOW_SIZE = 100.0


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def moving_average(data):
    return [
        sum(data[int(i - MA_WINDOW_SIZE + 1) : i + 1]) / MA_WINDOW_SIZE
        if i >= MA_WINDOW_SIZE
        else sum(data[: i + 1]) * 1.0 / (i + 1)
        for i in range(len(data))
    ]


# pyre-fixme[3]: Return type must be annotated.
def main():
    logging.basicConfig(level=logging.DEBUG)
    env = GymEnvironment("CartPole-v1")

    assert len(sys.argv) <= 2, "Usage: <figure_gen> <num_episodes>"
    num_episodes = 1000  # default value
    if len(sys.argv) == 2:
        num_episodes = int(sys.argv[1])

    agent = PearlAgent(
        policy_learner=DeepQLearning(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            hidden_dims=[64, 64],
            training_rounds=20,
        ),
        replay_buffer=FIFOOffPolicyReplayBuffer(10000),
    )
    info = online_learning(
        agent,
        env,
        number_of_episodes=num_episodes,
        learn_after_episode=True,
    )
    plt.plot(info["return"], label="vanilla dqn")
    plt.plot(moving_average(info["return"]), label="dqn_ma")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    agent = PearlAgent(
        policy_learner=DeepSARSA(
            env.observation_space.shape[0],
            env.action_space,
            # pyre-fixme[6]: For 3rd argument expected `Optional[ExplorationModule]`
            #  but got `List[int]`.
            [64, 64],
            training_rounds=20,
        ),
        replay_buffer=FIFOOnPolicyReplayBuffer(10000),
    )
    info = online_learning(
        agent,
        env,
        number_of_episodes=num_episodes,
        learn_after_episode=True,
    )
    plt.plot(info["return"], label="sarsa")
    plt.plot(moving_average(info["return"]), label="sarsa_ma")
    plt.legend()
    plt.savefig("figure_gen.png")


if __name__ == "__main__":
    main()
