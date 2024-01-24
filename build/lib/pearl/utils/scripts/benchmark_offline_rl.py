# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
from typing import List, Optional

import torch

from pearl.api.agent import Agent
from pearl.api.environment import Environment

from pearl.neural_networks.sequential_decision_making.actor_networks import (
    VanillaContinuousActorNetwork,
)

from pearl.pearl_agent import PearlAgent

from pearl.policy_learners.sequential_decision_making.implicit_q_learning import (
    ImplicitQLearning,
)

from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)

from pearl.utils.functional_utils.experimentation.create_offline_data import (
    create_offline_data,
    get_data_collection_agent_returns,
)
from pearl.utils.functional_utils.train_and_eval.offline_learning_and_evaluation import (
    get_offline_data_in_buffer,
    offline_evaluation,
    offline_learning,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment


def get_random_agent_returns(
    agent: Agent,
    env: Environment,
    save_path: Optional[str],
    file_path: Optional[str] = None,
    learn: bool = False,
    learn_after_episode: bool = False,
    evaluation_episodes: int = 500,
    seed: Optional[int] = None,
) -> List[float]:

    """
    This function returns a list of episode returns of a Pearl Agent interacting with the input
    environment using a randomly instantiated policy learner. This is needed to compute
    the baseline for calculating  normalized scores for offline rl benchmarks.


    Args:
        agent: a pearl agent with a randomly initiated policy learner.
        env: an environment to collect data from (e.g. GymEnvironment)
        learn: should be set to False.
        exploit: should be set to True.
        learn_after_episode: should be set to False.
    """

    # evaluation results of a random agent (with no learning)
    print("getting returns of a random agent")

    # check for a pickle file with episodic returns saved in the file path
    if file_path is not None:
        if os.path.isfile(file_path):
            print(f"loading returns from file {file_path}")
            with open(file_path, "rb") as file:
                # @lint-ignore PYTHONPICKLEISBAD
                random_agent_returns = pickle.load(file)
        else:
            raise FileNotFoundError(f"No file found at {file_path}")
    else:
        print(
            "no returns file path provided; proceeding to collect data from environment directly"
        )
        random_agent_returns = []
        for i in range(evaluation_episodes):
            evaluation_seed = seed + i if seed is not None else None
            info, _ = run_episode(
                agent=agent,
                env=env,
                learn=False,
                exploit=True,
                learn_after_episode=False,
                seed=evaluation_seed,
            )
            g = info["return"]
            print(f"episode {i}, return={g}")
            random_agent_returns.append(g)

            with open(
                save_path + "returns_random_agent" + ".pickle",
                "wb",
            ) as handle:
                # @lint-ignore PYTHONPICKLEISBAD
                pickle.dump(
                    random_agent_returns, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

    return random_agent_returns


# Can be generalized to different environment types; written for gym tasks for now
def evaluate_offline_rl(
    env: GymEnvironment,
    is_action_continuous: bool,
    offline_agent: PearlAgent,
    method_name: str,
    training_epochs: int = 1000,
    evaluation_episodes: int = 500,
    url: Optional[str] = None,
    data_path: Optional[str] = None,
    data_collection_agent: Optional[PearlAgent] = None,
    file_name: Optional[str] = None,
    data_save_path: Optional[str] = "offline_rl_data/",
    data_size: int = 1000000,
    seed: Optional[int] = None,
) -> List[float]:

    """
    This function trains and evaluates an offline RL agent on the given environment. Training data
    can be provided through a url or by specifying a local file path. If neither are provided,
    then a 'data collection agent' must be provided. The data collection agent will be used to
    collect data from the environment and save it to a file.

    Args:
        env: an environment to (optionally) collect data from (e.g. GymEnvironment) and evalue the
            offline agent on.
        is_action_continuous: used when translating the offline data to a replay buffer format
            compatible with a Pearl Agent (see class TensorBasedReplayBuffer for details).
        offline_agent: an offline agent to train and evaluate (for example, IQL or CQL based agent).
        method_name: name of the agent's policy learner (used for saving results).
        training_epochs: number of epochs to train the offline agent for.
        evaluation_episodes: number of episodes to evaluate the offline agent for.
        url: url to download data from.
        data_path: path to a local file containing offline data to use for training.
        data_collection_agent: a Pearl Agent used to collect offline data in case url or data_path
            are not provided.
        file_name: name of the file to store the offline data in.
        data_save_path: path to a directory where the offline data will be stored.
        data_size: size of the offline data (replay buffer) to be used for training.
    """

    if url is None and data_path is None and data_collection_agent is None:
        raise ValueError(
            "Must provide either a URL, a path, or an agent to collect data."
        )

    # to save the offline evaluation results and/or offline data collected (if no url or data path
    # is provided)
    os.makedirs(data_save_path, exist_ok=True)

    if url is not None or data_path is not None:
        if url is not None:
            print("downloading data from the given url")
        else:
            if os.path.isfile(data_path):
                print("reading data from the given path")
            else:
                raise FileNotFoundError(f"No file found at {data_path}")

        offline_data_replay_buffer = get_offline_data_in_buffer(
            is_action_continuous, url, data_path, size=data_size
        )
    else:
        if file_name is None:
            raise ValueError("Must provide a name of file to store data.")

        file_name = file_name
        create_offline_data(
            agent=data_collection_agent,
            env=env,
            save_path=data_save_path,
            file_name=file_name,
            max_len_offline_data=data_size,
            learn=True,
            learn_after_episode=False,
            seed=seed,
        )
        print("\n")
        print("collected data; starting import in replay buffer")
        data_path = data_save_path + file_name
        offline_data_replay_buffer = get_offline_data_in_buffer(
            is_action_continuous, url, data_path, size=data_size
        )

    print("offline data in replay buffer; start offline training")
    offline_learning(
        offline_agent=offline_agent,
        data_buffer=offline_data_replay_buffer,
        training_epochs=training_epochs,
        seed=seed,
    )

    print("\n")
    print("offline training done; start offline evaluation")
    offline_evaluation_returns = offline_evaluation(
        offline_agent=offline_agent,
        env=env,
        number_of_episodes=evaluation_episodes,
        seed=seed,
    )

    # save the offline evaluation results
    dir_name = (
        data_save_path
        + method_name
        + "/"
        + offline_agent.policy_learner._actor.__class__.__name__
        + "/"
    )
    os.makedirs(dir_name, exist_ok=True)

    with open(
        dir_name
        + "returns_offline_agent_"
        + dataset
        + "_"
        + str(training_epochs)
        + ".pickle",
        "wb",
    ) as handle:
        # @lint-ignore PYTHONPICKLEISBAD
        pickle.dump(
            offline_evaluation_returns, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    return offline_evaluation_returns


if __name__ == "__main__":
    device_id = 1  # use -1 for CPU, 0, 1, for cuda
    experiment_seed = 100
    env_name = "HalfCheetah-v4"
    env = GymEnvironment(env_name)
    action_space = env.action_space
    is_action_continuous = True

    actor_network_type = VanillaContinuousActorNetwork

    offline_agent = PearlAgent(
        policy_learner=ImplicitQLearning(
            state_dim=env.observation_space.shape[0],  # pyre-ignore
            action_space=action_space,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            value_critic_hidden_dims=[256, 256],
            actor_network_type=actor_network_type,
            value_critic_learning_rate=1e-4,
            actor_learning_rate=3e-4,
            critic_learning_rate=1e-4,
            critic_soft_update_tau=0.05,
            training_rounds=2,
            batch_size=256,
            expectile=0.75,
            temperature_advantage_weighted_regression=3,
        ),
        device_id=device_id,
    )

    data_collection_agent = PearlAgent(
        policy_learner=ContinuousSoftActorCritic(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            training_rounds=1,
            batch_size=256,
            entropy_coef=0.25,
            entropy_autotune=False,
            actor_learning_rate=0.0003,
            critic_learning_rate=0.0005,
        ),
        replay_buffer=FIFOOffPolicyReplayBuffer(1000000),
        device_id=device_id,
    )

    data_save_path = "offline_rl_data/" + env_name + "/"
    # dataset = "small_2"
    # dataset = "medium"

    # this is only for end to end testing check
    # for benchmarking, using the "small_2", "medium" or "large" datasets
    dataset = "small"
    file_name = "offline_raw_transitions_dict_" + dataset + ".pt"

    print(" ")
    print(
        f"actor network type: {offline_agent.policy_learner._actor.__class__.__name__}"
    )
    print(f"data set name: {file_name}")
    print(
        f"critic update parameter: {offline_agent.policy_learner._critic_soft_update_tau}"
    )
    print(" ")

    data_path = data_save_path + file_name

    # remember to specify a data collection agent or a path to offline data
    offline_agent_returns = evaluate_offline_rl(
        env=env,
        is_action_continuous=is_action_continuous,
        offline_agent=offline_agent,
        method_name="Implicit Q learning",
        training_epochs=10000,
        # data_path=data_path,
        data_collection_agent=data_collection_agent,
        file_name=file_name,
        data_save_path=data_save_path,
        data_size=100000,
        evaluation_episodes=100,
        seed=experiment_seed,
    )
    avg_offline_agent_returns = torch.mean(torch.tensor(offline_agent_returns))
    print()
    print(f"average returns of the offline agent {avg_offline_agent_returns}")
    print("\n")

    # getting the returns of a random agent
    random_returns_file_path = data_save_path + "returns_random_agent.pickle"
    random_agent_returns = get_random_agent_returns(
        agent=data_collection_agent,
        env=env,
        save_path=data_save_path,
        file_path=random_returns_file_path,
    )
    avg_return_random_agent = torch.mean(torch.tensor(random_agent_returns))
    print(f"average returns of a random agent {avg_return_random_agent}")
    print("\n")

    data_collection_agent_returns = get_data_collection_agent_returns(
        data_path=data_path
    )

    avg_return_data_collection_agent = torch.mean(
        torch.tensor(data_collection_agent_returns)
    )
    print(
        f"average returns of the data collection agent {avg_return_data_collection_agent}"
    )
    print(" ")

    normalized_score = (avg_offline_agent_returns - avg_return_random_agent) / (
        avg_return_data_collection_agent - avg_return_random_agent
    )

    # ideally, we would want normalized score to be greater than 1 (indicating the agent has
    # learned something better than the data collection agent) but this is not always the case
    print(f"normalized score {normalized_score}")
    if normalized_score < 0.25:
        print(
            "Offline agent does not seems to be learning well. Check the "
            "hyperparameters in IQL_offline_method in benchmark_config.py file "
            "and run with the dataset_name = `small_2`."
        )
