# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import pickle
from collections import deque

import torch
from pearl.api.environment import Environment
from pearl.api.reward import Value
from pearl.pearl_agent import PearlAgent
from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode


def create_offline_data(
    agent: PearlAgent,
    env: Environment,
    save_path: str,
    file_name: str,
    max_len_offline_data: int = 50000,
    learn: bool = True,
    exploit: bool = False,
    learn_after_episode: bool = True,
    evaluation_episodes: int = 100,
    seed: int | None = None,
) -> list[Value]:
    """
    This function creates offline data by interacting with a given environment using a specified
    agent. This is mostly for illustration with standard benchmark environments. For most
    practical use cases, offline data collection will use custom pipelines.

    Transition tuples are stored in .pt file in the specified path. Training returns
    (episodic returns during training) of the agent are saved in a pickle file. At the end of data
    collection, evaluation returns of the final agent are also saved in a pickle file. This
    approximates the performance of the best learned policy in the offline data.

    Note: Much of this function overlaps with `run_episode` function in
        `pearl/utils/functional_utils/train_and_eval/online_learning.py`.

    Args:
        agent (PearlAgent): A pearl agent with policy learner, exploration module and replay buffer
            specified. For e.g. a DQN agent.
        env (Environment): An environment to collect data from (e.g. `GymEnvironment`).
        save_path (str): Path to save the offline data.
        file_name (str): Name of the file to save the raw transition tuples.
        max_len_offline_data (int): Number of the transition tuples to be collected in offline data.
        learn (bool): When set to True, the agent learns after each environment interaction.
            Defaults to True.
        exploit (bool): When set to True, the agent does not explore and acts greedily with respect
            to the current estimate of the optimal policy. Defaults to False as we want the agent
            to explore during data collection for standard benchmarks.
        learn_after_episode (bool): When set to True, the agent learns after each episode.
            Defaults to False.
        evaluation_episodes (int): The number of episodes to evaluate the trained agent on.
            Defaults to 100.
        seed (int, optional): Environment seed for reproducibility.

    Returns:
        returns_offline_agent: a list of returns for each evaluation episode.
    """

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
            # exploit is explicitly set to False as we want exploration during data collection with
            # standard benchmark environments like Gym, MuJoCo etc.
            action = agent.act(exploit=False)

            action_result = env.step(action)
            # pyre-fixme[58]: `+` is not supported for operand types `int` and `object`.
            g += action_result.reward
            agent.observe(action_result)
            transition_tuple = {
                "observation": observation,
                "action": action,
                "reward": action_result.reward,
                "next_observation": action_result.observation,
                "curr_available_actions": env.action_space,
                "next_available_actions": env.action_space,
                "terminated": action_result.terminated,
                "truncated": action_result.truncated,
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
    returns_file_path: str | None = None,
) -> list[Value]:
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
            data = torch.load(
                file, map_location=torch.device("cpu"), weights_only=False
            )

        data_collection_agent_returns = []
        g = 0
        for transition in list(data):
            if transition["terminated"] or transition["truncated"]:
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
