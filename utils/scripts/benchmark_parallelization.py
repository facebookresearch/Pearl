#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors

"""
The code in this file provides a way to run multiple pearl experiments in different
processes using torch.multiprocessing.
Outputs of the code are saved in the folder ~/pearl_execution/outputs/.
To run the code, enter the pearl directory, then run
./utils/scripts/meta_only/run_pearl.sh utils/scripts/benchmark_parallelization.py
(make sure conda and related packages have been installed with
./utils/scripts/meta_only/setup_conda_pearl_on_devserver.sh)
"""
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (  # noqa E501
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (  # noqa E501
    NormalDistributionExploration,
)
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
)
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (  # noqa E501
    ContinuousSoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.td3 import TD3
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (  # noqa E501
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.on_policy_episodic_replay_buffer import (  # noqa E501
    OnPolicyEpisodicReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import (
    online_learning_returns,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

warnings.filterwarnings("ignore")


def run(experiments) -> None:
    """Assign one run to one process."""
    assert len(experiments) > 0
    all_processes = []

    for e in experiments:
        evaluate(e, all_processes)

    for p in all_processes:
        p.start()
        p.join()


def evaluate(experiment, all_processes: List[mp.Process]) -> None:
    """Running multiple methods and multiple runs in the given gym environment."""
    env_name = experiment["env_name"]
    num_runs = experiment["num_runs"]
    num_episodes = experiment["num_episodes"]
    print_every_x_episodes = experiment["print_every_x_episodes"]
    methods = experiment["methods"]
    processes = []
    for method in methods:
        for run_idx in range(num_runs):
            p = mp.Process(
                target=evaluate_single,
                args=(
                    env_name,
                    method,
                    run_idx,
                    num_episodes,
                    print_every_x_episodes,
                ),
            )
            processes.append(p)

    all_processes.extend(processes)


def evaluate_single(
    env_name,
    method,
    run_idx,
    num_episodes,
    print_every_x_episodes,
):
    """Performing one run of experiment."""
    policy_learner = method["policy_learner"]
    policy_learner_args = method["policy_learner_args"]
    agent_args = method["agent_args"]
    env = GymEnvironment(env_name)
    if "exploration_module" in method and "exploration_module_args" in method:
        policy_learner_args["exploration_module"] = method["exploration_module"](
            **method["exploration_module_args"]
        )
    if "replay_buffer" in method and "replay_buffer_args" in method:
        agent_args["replay_buffer"] = method["replay_buffer"](
            **method["replay_buffer_args"]
        )
    agent = PearlAgent(
        policy_learner=policy_learner(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            **policy_learner_args,
        ),
        **agent_args,
    )
    method_name = method["name"]
    print(f"Run #{run_idx + 1} for {method_name} in {env_name}")

    returns = online_learning_returns(
        agent,
        env,
        number_of_episodes=num_episodes,
        print_every_x_episodes=print_every_x_episodes,
    )
    dir = f"outputs/{env_name}/{method_name}"
    os.makedirs(dir, exist_ok=True)
    np.save(dir + f"/{run_idx}.npy", returns)


def generate_plots(experiments) -> None:
    for e in experiments:
        generate_one_plot(e)


def generate_one_plot(experiment):
    """Generating learning curves for all tested methods in one environment."""
    env_name = experiment["env_name"]
    num_runs = experiment["num_runs"]
    methods = experiment["methods"]
    for method in methods:
        data = []
        for run in range(num_runs):
            try:
                d = np.load(f"outputs/{env_name}/{method['name']}/{run}.npy")
            except FileNotFoundError:
                print(
                    f"File not found for outputs/{env_name}/{method['name']}/{run}.npy"
                )
                continue
            data.append(d)
        data = np.array(data)
        mean = data.mean(axis=0)
        std_error = data.std(axis=0) / np.sqrt(num_runs)
        plt.plot(np.arange(mean.shape[0]), mean, label=method["name"])
        plt.fill_between(
            np.arange(mean.shape[0]), mean - std_error, mean + std_error, alpha=0.2
        )
    plt.title(env_name)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(f"outputs/{env_name}.png")
    plt.close()


if __name__ == "__main__":
    experiments = [
        {
            "env_name": "CartPole-v0",
            "num_runs": 5,
            "num_episodes": 500,
            "print_every_x_episodes": 10,
            "methods": [
                {
                    "name": "DQN",
                    "policy_learner": DeepQLearning,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64],
                        "training_rounds": 20,
                        "batch_size": 64,
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": EGreedyExploration,
                    "exploration_module_args": {"epsilon": 0.1},
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 10000},
                },
                {
                    "name": "PPO",
                    "policy_learner": ProximalPolicyOptimization,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64],
                        "training_rounds": 50,
                        "batch_size": 64,
                        "epsilon": 0.1,
                    },
                    "agent_args": {"device_id": -1},
                    "replay_buffer": OnPolicyEpisodicReplayBuffer,
                    "replay_buffer_args": {"capacity": 10000},
                },
            ],
        },
        {
            "env_name": "Acrobot-v1",
            "num_runs": 5,
            "num_episodes": 500,
            "print_every_x_episodes": 10,
            "methods": [
                {
                    "name": "DQN",
                    "policy_learner": DeepQLearning,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64],
                        "training_rounds": 20,
                        "batch_size": 64,
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": EGreedyExploration,
                    "exploration_module_args": {"epsilon": 0.1},
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 10000},
                },
                {
                    "name": "PPO",
                    "policy_learner": ProximalPolicyOptimization,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64],
                        "training_rounds": 50,
                        "batch_size": 64,
                        "epsilon": 0.1,
                    },
                    "agent_args": {"device_id": -1},
                    "replay_buffer": OnPolicyEpisodicReplayBuffer,
                    "replay_buffer_args": {"capacity": 10000},
                },
            ],
        },
        {
            "env_name": "Pendulum-v1",
            "num_runs": 5,
            "num_episodes": 500,
            "print_every_x_episodes": 10,
            "methods": [
                {
                    "name": "DDPG",
                    "policy_learner": DeepDeterministicPolicyGradient,
                    "policy_learner_args": {
                        "hidden_dims": [400, 300],
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": NormalDistributionExploration,
                    "exploration_module_args": {
                        "mean": 0,
                        "std_dev": 0.2,
                        "max_action_value": 2,
                        "min_action_value": -2,
                    },
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 50000},
                },
                {
                    "name": "TD3",
                    "policy_learner": TD3,
                    "policy_learner_args": {
                        "hidden_dims": [400, 300],
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": NormalDistributionExploration,
                    "exploration_module_args": {
                        "mean": 0,
                        "std_dev": 0.2,
                        "max_action_value": 2,
                        "min_action_value": -2,
                    },
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 50000},
                },
                {
                    "name": "ContinuousSAC",
                    "policy_learner": ContinuousSoftActorCritic,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64, 64],
                        "training_rounds": 1,
                        "batch_size": [256],
                        "entropy_coef": 0.1,
                        "actor_learning_rate": 0.0005,
                        "critic_learning_rate": 0.0005,
                    },
                    "agent_args": {"device_id": -1},
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 50000},
                },
            ],
        },
        {
            "env_name": "HalfCheetah-v4",
            "num_runs": 5,
            "num_episodes": 500,
            "print_every_x_episodes": 10,
            "methods": [
                {
                    "name": "DDPG",
                    "policy_learner": DeepDeterministicPolicyGradient,
                    "policy_learner_args": {
                        "hidden_dims": [256, 256],
                        "critic_learning_rate": 1e-3,
                        "actor_learning_rate": 1e-3,
                        "training_rounds": 1,
                        "batch_size": 256,
                        "discount_factor": 0.99,
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": NormalDistributionExploration,
                    "exploration_module_args": {
                        "mean": 0,
                        "std_dev": 0.1,
                        "max_action_value": 1,
                        "min_action_value": -1,
                    },
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 1000000},
                },
                {
                    "name": "TD3",
                    "policy_learner": TD3,
                    "policy_learner_args": {
                        "hidden_dims": [256, 256],
                        "critic_learning_rate": 1e-3,
                        "actor_learning_rate": 1e-3,
                        "training_rounds": 1,
                        "batch_size": 256,
                        "discount_factor": 0.99,
                        "learning_action_noise_std": 0.2,
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": NormalDistributionExploration,
                    "exploration_module_args": {
                        "mean": 0,
                        "std_dev": 0.1,
                        "max_action_value": 1,
                        "min_action_value": -1,
                    },
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 1000000},
                },
                {
                    "name": "ContinuousSAC",
                    "policy_learner": ContinuousSoftActorCritic,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64, 64],
                        "training_rounds": 1,
                        "batch_size": [256],
                        "entropy_coef": 0.1,
                        "actor_learning_rate": 0.0005,
                        "critic_learning_rate": 0.0005,
                    },
                    "agent_args": {"device_id": -1},
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 50000},
                },
            ],
        },
        {
            "env_name": "Ant-v4",
            "num_runs": 5,
            "num_episodes": 500,
            "print_every_x_episodes": 10,
            "methods": [
                {
                    "name": "DDPG",
                    "policy_learner": DeepDeterministicPolicyGradient,
                    "policy_learner_args": {
                        "hidden_dims": [256, 256],
                        "critic_learning_rate": 1e-3,
                        "actor_learning_rate": 1e-3,
                        "training_rounds": 1,
                        "batch_size": 256,
                        "discount_factor": 0.99,
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": NormalDistributionExploration,
                    "exploration_module_args": {
                        "mean": 0,
                        "std_dev": 0.1,
                        "max_action_value": 1,
                        "min_action_value": -1,
                    },
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 1000000},
                },
                {
                    "name": "TD3",
                    "policy_learner": TD3,
                    "policy_learner_args": {
                        "hidden_dims": [256, 256],
                        "critic_learning_rate": 1e-3,
                        "actor_learning_rate": 1e-3,
                        "training_rounds": 1,
                        "batch_size": 256,
                        "discount_factor": 0.99,
                        "learning_action_noise_std": 0.2,
                    },
                    "agent_args": {"device_id": -1},
                    "exploration_module": NormalDistributionExploration,
                    "exploration_module_args": {
                        "mean": 0,
                        "std_dev": 0.1,
                        "max_action_value": 1,
                        "min_action_value": -1,
                    },
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 1000000},
                },
                {
                    "name": "ContinuousSAC",
                    "policy_learner": ContinuousSoftActorCritic,
                    "policy_learner_args": {
                        "hidden_dims": [64, 64, 64],
                        "training_rounds": 1,
                        "batch_size": [256],
                        "entropy_coef": 0.1,
                        "actor_learning_rate": 0.0005,
                        "critic_learning_rate": 0.0005,
                    },
                    "agent_args": {"device_id": -1},
                    "replay_buffer": FIFOOffPolicyReplayBuffer,
                    "replay_buffer_args": {"capacity": 50000},
                },
            ],
        },
    ]
    run(experiments)
    generate_plots(experiments)
