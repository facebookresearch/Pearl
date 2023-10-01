#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from collections import deque

import torch
from pearl.api.agent import Agent
from pearl.api.environment import Environment
from pearl.utils.functional_utils.experimentation.set_seed import set_seed


def create_offline_data(
    agent: Agent,
    env: Environment,
    max_len_offline_data: int = 50000,
    # pyre-fixme[2]: Parameter must be annotated.
    learn=True,
    # pyre-fixme[2]: Parameter must be annotated.
    learn_after_episode=True,
) -> None:

    """
    This function creates offline data by interacting with a given environment using a
    reinforcement learning algorithm. This is mostly for illustration with standard
    benchmark environments only. For most practical use cases, offline data collection
    will use custom pipelines.

    Args:
        agent: a pearl agent with policy learner, exploration module and replay buffer specified (e.g. a DQN agent).
        env: an environment to collect data from (e.g. GymEnvironment)
        number_of_episodes: number of episodes for which data is to be collected.
        learn: whether to learn after each episode (depends on the policy learner used by agent).
        exploit: set as default to False as we want exploration during data collection.
        learn_after_episode: whether to learn after each episode (depends on the policy learner used by agent).
    """

    # much of this function overlaps with episode return function but i think writing it
    # like this is cleaner

    set_seed(111)
    epi_returns = []
    epi = 0
    raw_transitions_buffer = deque([], maxlen=max_len_offline_data)
    while len(raw_transitions_buffer) < max_len_offline_data:
        if epi % 100 == 0:
            print("episode", epi)
        g = 0
        observation, action_space = env.reset()
        agent.reset(observation, action_space)
        done = False
        while not done:
            # pyre-fixme[28]: Unexpected keyword argument `exploit`.
            action = agent.act(
                exploit=False
            )  # exploit is explicitly set to False as we want exploration during data collection.
            action_result = env.step(action)
            g += action_result.reward
            agent.observe(action_result)
            transition_tuple = {
                "observation": observation,
                "action": action,
                "reward": action_result.reward,
                "next_observation": action_result.observation,
                "curr_available_actions": env.action_space,
                "next_available_actions": env.action_space,
                "action_space": env.action_space,
                "done": action_result.done,
            }
            raw_transitions_buffer.append(transition_tuple)
            if learn and not learn_after_episode:
                agent.learn()
            done = action_result.done

        if learn and learn_after_episode:
            agent.learn()

        epi_returns.append(g)
        epi += 1

    torch.save(raw_transitions_buffer, "offline_raw_transitions_dict.pt")

    # pyre-fixme[7]: Expected `None` but got `List[typing.Any]`.
    return epi_returns  # for plotting; to check the quality of the policy used to collect offine data
