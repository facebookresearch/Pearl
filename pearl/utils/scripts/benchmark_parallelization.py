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

from pearl.utils.functional_utils.train_and_eval.online_learning import (
    online_learning_returns,
)
from pearl.utils.scripts.benchmark_config import (
    # all_ac_discrete_control_methods,
    # all_continuous_control_methods,
    # ple_steps,
    classic_control_steps,
    get_env,
    # SARSA_method,
    # TD3_method,
    # all_discrete_control_methods,
    # BootstrappedDQN_method,
    # CDQN_method,
    # CSAC_method,
    # DDPG_method,
    # DQN_method,
    IQL_method,
    # all_partial_observable_continuous_control_envs,
    # DDQN_method,
    # mujoco_envs,
    # all_continuous_control_envs,
    # all_discrete_control_envs,
    # all_partial_observable_discrete_control_envs,
    # all_safety_discrete_control_envs,
    # all_sparse_reward_continuous_control_envs,
    # all_sparse_reward_discrete_control_envs,
    # classic_continuous_control_envs,
    # mujoco_steps,
    num_runs,
    # DuelingDQN_method,
    PPO_method,
    print_every_x_steps,
    # QRDQN_method,
    REINFORCE_method,
    SAC_method,
)

warnings.filterwarnings("ignore")


def run(experiments) -> None:
    """Assign one run to one process."""
    assert len(experiments) > 0
    all_processes = []

    for e in experiments:
        evaluate(e, all_processes)

    for p in all_processes:
        p.start()
    for p in all_processes:
        p.join()


def evaluate(experiment, all_processes: List[mp.Process]) -> None:
    """Running multiple methods and multiple runs in the given gym environment."""
    env_name = experiment["env_name"]
    num_runs = experiment["num_runs"]
    num_episodes = experiment.get("num_episodes")
    num_steps = experiment.get("num_steps")
    print_every_x_episodes = experiment.get("print_every_x_episodes")
    print_every_x_steps = experiment.get("print_every_x_steps")
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
                    num_steps,
                    print_every_x_episodes,
                    print_every_x_steps,
                ),
            )
            processes.append(p)

    all_processes.extend(processes)


def evaluate_single(
    env_name,
    method,
    run_idx,
    num_episodes,
    num_steps,
    print_every_x_episodes,
    print_every_x_steps,
):
    """Performing one run of experiment."""
    policy_learner = method["policy_learner"]
    policy_learner_args = method["policy_learner_args"]
    agent_args = method["agent_args"]
    env = get_env(env_name)
    if "exploration_module" in method and "exploration_module_args" in method:
        policy_learner_args["exploration_module"] = method["exploration_module"](
            **method["exploration_module_args"]
        )
    if "replay_buffer" in method and "replay_buffer_args" in method:
        agent_args["replay_buffer"] = method["replay_buffer"](
            **method["replay_buffer_args"]
        )
    if "safety_module" in method and "safety_module_args" in method:
        agent_args["safety_module"] = method["safety_module"](
            **method["safety_module_args"]
        )
    if (
        "action_representation_module" in method
        and "action_representation_module_args" in method
    ):
        if (
            method["action_representation_module"].__name__
            == "OneHotActionTensorRepresentationModule"
        ):
            method["action_representation_module_args"][
                "max_actions"
            ] = env.action_space.n
        agent_args["action_representation_module"] = method[
            "action_representation_module"
        ](**method["action_representation_module_args"])

    if method["name"] == "DuelingDQN":  # only for Dueling DQN
        assert "network_module" in method and "network_args" in method
        policy_learner_args["network_instance"] = method["network_module"](
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **method["network_args"],
        )
    if method["name"] == "BootstrappedDQN":  # only for Bootstrapped DQN
        assert "network_module" in method and "network_args" in method
        policy_learner_args["q_ensemble_network"] = method["network_module"](
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **method["network_args"],
        )
    else:
        policy_learner_args["state_dim"] = env.observation_space.shape[0]

    policy_learner_args["action_space"] = env.action_space
    agent = PearlAgent(
        policy_learner=policy_learner(
            **policy_learner_args,
        ),
        **agent_args,
    )
    method_name = method["name"]
    print(f"Run #{run_idx + 1} for {method_name} in {env_name}")
    if (
        method["name"] == "REINFORCE" or method["name"] == "PPO"
    ):  # REINFORCE only performs learning at the end of each episode
        learn_after_episode = True
    else:
        learn_after_episode = False

    returns = online_learning_returns(
        agent,
        env,
        number_of_episodes=num_episodes,
        number_of_steps=num_steps,
        print_every_x_episodes=print_every_x_episodes,
        print_every_x_steps=print_every_x_steps,
        learn_after_episode=learn_after_episode,
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
    methods = [IQL_method]
    envs = ["CartPole-v0"]
    # envs = ["CartPole-v0"]
    num_steps = classic_control_steps
    experiments = [
        {
            "env_name": env_name,
            "num_runs": num_runs,
            "num_steps": num_steps,
            "print_every_x_steps": print_every_x_steps,
            "methods": methods,
        }
        for env_name in envs
    ]
    run(experiments)
    generate_plots(experiments)
