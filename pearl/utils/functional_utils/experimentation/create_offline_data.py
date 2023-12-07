# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
from collections import deque
from typing import List, Optional

import torch
from pearl.api.environment import Environment
from pearl.api.reward import Value
from pearl.pearl_agent import PearlAgent
from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


def create_offline_data(
    agent: PearlAgent,
    env: Environment,
    save_path: str,
    file_name: str,
    max_len_offline_data: int = 50000,
    learn: bool = True,
    learn_after_episode: bool = True,
    evaluation_episodes: int = 100,
    seed: Optional[int] = None,
) -> List[Value]:

    """
    This function creates offline data by interacting with a given environment using a specified
    Pearl Agent. This is mostly for illustration with standard benchmark environments only.
    For most practical use cases, offline data collection will use custom pipelines.

    Args:
        agent: a pearl agent with policy learner, exploration module and replay buffer specified
               (e.g. a DQN agent).
        env: an environment to collect data from (e.g. GymEnvironment)
        number_of_episodes: number of episodes for which data is to be collected.
        learn: whether to learn after each episode (depends on the policy learner used by agent).
        exploit: set as default to False as we want exploration during data collection.
        learn_after_episode: whether to learn after each episode
        (depends on the policy learner used by agent).
    """

    # much of this function overlaps with episode return function but i think writing it
    # like this is cleaner

    print(f"collecting data from env: {env} using agent: {agent}")

    epi_returns = []
    epi = 0
    raw_transitions_buffer = deque([], maxlen=max_len_offline_data)
    while len(raw_transitions_buffer) < max_len_offline_data:
        g = 0
        observation, action_space = env.reset(seed=seed)
        agent.reset(observation, action_space)
        done = False
        while not done:
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
                "done": action_result.done,
                "max_number_actions": env.action_space.n
                if isinstance(env.action_space, DiscreteActionSpace)
                else None,
            }

            observation = action_result.observation
            raw_transitions_buffer.append(transition_tuple)
            if learn and not learn_after_episode:
                agent.learn()
            done = action_result.done

        if learn and learn_after_episode:
            agent.learn()

        epi_returns.append(g)
        print(f"\rEpisode {epi}, return={g}", end="")
        epi += 1

    # save offline transition tuples in a .pt file
    torch.save(raw_transitions_buffer, save_path + file_name)

    # save training returns of the data collection agent
    with open(
        save_path
        + "training_returns_data_collection_agent_"
        + str(max_len_offline_data)
        + ".pickle",
        "wb",
    ) as handle:
        # @lint-ignore PYTHONPICKLEISBAD
        pickle.dump(epi_returns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # evaluation results of the data collection agent
    print(" ")
    print(
        "data collection complete; starting evaluation runs for data collection agent"
    )

    evaluation_returns = []
    for i in range(evaluation_episodes):
        # data creation and evaluation seed should be different
        evaluation_seed = seed + i if seed is not None else seed
        episode_info, _ = run_episode(
            agent=agent,
            env=env,
            learn=False,
            exploit=True,
            learn_after_episode=False,
            seed=evaluation_seed,
        )
        g = episode_info["return"]
        print(f"\repisode {i}, return={g}", end="")
        evaluation_returns.append(g)

    with open(
        save_path
        + "evaluation_returns_data_collection_agent_"
        + str(max_len_offline_data)
        + ".pickle",
        "wb",
    ) as handle:
        # @lint-ignore PYTHONPICKLEISBAD
        pickle.dump(evaluation_returns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return epi_returns  # for plotting returns of the policy used to collect offine data


# getting returns of the data collection agent, either from file or by stitching trajectories
# in the training data
def get_data_collection_agent_returns(
    data_path: str,
    returns_file_path: Optional[str] = None,
) -> List[Value]:

    """
    This function returns episode returns of a Pearl Agent using for offline data collection.
    The returns file can be directly provided or we can stitch together trajectories in the offline
    data. This function is used to compute normalized scores for offline rl benchmarks.

    Args:
        data_path: path to the directory where the offline data is stored.
        returns_file_path: path to the file containing returns of the data collection agent.
    """

    print("getting returns of the data collection agent agent")
    if returns_file_path is None:
        print(
            f"using offline training data in {data_path} to stitch trajectories and compute returns"
        )
        with open(data_path, "rb") as file:
            data = torch.load(file, map_location=torch.device("cpu"))

        data_collection_agent_returns = []
        g = 0
        for transition in list(data):
            if transition["done"]:
                data_collection_agent_returns.append(g)
                g = 0
            else:
                g += transition["reward"]
    else:
        print(f"loading returns from file {returns_file_path}")
        with open(returns_file_path, "rb") as file:
            # @lint-ignore PYTHONPICKLEISBAD
            data_collection_agent_returns = pickle.load(file)

    return data_collection_agent_returns
