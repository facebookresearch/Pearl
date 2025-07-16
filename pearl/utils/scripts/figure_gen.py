# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
import sys

import matplotlib.pyplot as plt
from pearl.api.reward import Value
from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.deep_sarsa import DeepSARSA
from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.sequential_decision_making.sarsa_replay_buffer import (
    SARSAReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

MA_WINDOW_SIZE = 100.0


def moving_average(data: list[Value]) -> Value:
    return [
        (
            # pyre-fixme[6]: For 1st argument expected `Iterable[Union[Literal[-20], ...
            sum(data[int(i - MA_WINDOW_SIZE + 1) : i + 1]) / MA_WINDOW_SIZE
            if i >= MA_WINDOW_SIZE
            # pyre-fixme[6]: For 1st argument expected `Iterable[Union[typing_extensi...
            else sum(data[: i + 1]) * 1.0 / (i + 1)
        )
        for i in range(len(data))
    ]


def main() -> None:
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
        replay_buffer=BasicReplayBuffer(1000),
    )
    info = online_learning(
        agent,
        env,
        number_of_episodes=num_episodes,
        learn_after_episode=True,
        print_every_x_episodes=int(num_episodes / 25),
    )
    plt.plot(info["return"], label="DQN")
    plt.plot(moving_average(info["return"]), label="DQN moving average")
    plt.xlabel("Episode")
    plt.ylabel("Return")

    agent = PearlAgent(
        policy_learner=DeepSARSA(
            env.observation_space.shape[0],
            env.action_space,
            hidden_dims=[64, 64],
            training_rounds=20,
        ),
        replay_buffer=SARSAReplayBuffer(1000),
    )
    info = online_learning(
        agent,
        env,
        number_of_episodes=num_episodes,
        learn_after_episode=True,
        print_every_x_episodes=int(num_episodes / 25),
    )
    plt.plot(info["return"], label="SARSA")
    plt.plot(moving_average(info["return"]), label="SARSA moving average")

    plt.legend()
    plt.savefig("figure_gen.png")


if __name__ == "__main__":
    main()
