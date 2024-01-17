# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.action_representation_modules.binary_action_representation_module import (
    BinaryActionTensorRepresentationModule,
)

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.policy_learner import PolicyLearner

from pearl.replay_buffers.contextual_bandits.discrete_contextual_bandit_replay_buffer import (
    DiscreteContextualBanditReplayBuffer,
)
from pearl.utils.instantiations.environments.contextual_bandit_uci_environment import (
    SLCBEnvironment,
)

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

from pearl.utils.scripts.cb_benchmark.cb_benchmark_config import (
    letter_uci_dict,
    pendigits_uci_dict,
    return_neural_fastcb_config,
    return_neural_lin_ts_config,
    return_neural_lin_ucb_config,
    return_neural_squarecb_config,
    return_offline_eval_config,
    run_config,
    satimage_uci_dict,
    yeast_uci_dict,
)
from pearl.utils.scripts.cb_benchmark.cb_download_benchmarks import download_uci_data


def online_evaluation(
    env: SLCBEnvironment,
    agent: PearlAgent,
    num_steps: int = 5000,
    training_mode: bool = True,
) -> List[float]:
    regrets = []
    for i in range(num_steps):
        observation, action_space = env.reset()
        agent.reset(observation, action_space)
        action = agent.act()
        regret = env.get_regret(action)
        action_result = env.step(action)
        if training_mode:
            agent.observe(action_result)
            agent.learn()
        regrets.append(regret)
        if i % 10 == 0:
            print("Step: ", i, " Avg Regret: ", sum(regrets) / len(regrets))
    return regrets


def train_via_uniform_data(
    env: SLCBEnvironment,
    agent: PearlAgent,
    T: int = 50000,
    training_epoches: int = 100,
    action_embeddings: str = "discrete",
) -> PearlAgent:
    """
    Get model trained on a dataset collected by acting with a uniform policy
    """
    for _ in range(T):
        observation, action_space = env.reset()
        assert isinstance(action_space, DiscreteActionSpace)
        agent.reset(observation, action_space)
        # take random action and add to the replay buffer
        coin_flip = random.choice([0, 1, 2, 3])
        if coin_flip == 0:
            action_ind = env._current_label
        else:
            action_ind = random.choice(range(action_space.n))
        agent._latest_action = env.action_transfomer(
            action_ind, action_embeddings=action_embeddings
        )

        # apply action to environment
        action_result = env.step(action_ind)
        agent.observe(action_result)

    agent.policy_learner.training_rounds = training_epoches * T
    agent.learn()

    return agent


def run_experiments_offline(
    env: SLCBEnvironment,
    T: int = 50000,
    training_rounds: int = 100,
    hidden_dims: Optional[List[int]] = None,
    num_eval_steps: int = 100,
    action_representation_module: Optional[ActionRepresentationModule] = None,
) -> List[float]:
    """
    Runs offline evaluation by training a `NeuralBandit` on the data collected
    by taking uniform actions.
    """
    if hidden_dims is None:
        hidden_dims = [64, 16]

    feature_dim = env.observation_dim
    dim_actions = env.bits_num

    if action_representation_module is None:
        action_representation_module = BinaryActionTensorRepresentationModule(
            bits_num=dim_actions
        )
    # prepare offline agent
    neural_greedy_policy = NeuralBandit(
        feature_dim=feature_dim + dim_actions,
        hidden_dims=hidden_dims,
        learning_rate=0.01,
        batch_size=128,
        training_rounds=T,
        exploration_module=NoExploration(),
        action_representation_module=action_representation_module,
    )

    agent = PearlAgent(
        policy_learner=neural_greedy_policy,
        replay_buffer=DiscreteContextualBanditReplayBuffer(T),
    )

    # training_epoches is set to be equal to training_rounds
    agent = train_via_uniform_data(env, agent, T=T, training_epoches=training_rounds)
    regrets = online_evaluation(
        env, agent, num_steps=num_eval_steps, training_mode=False
    )

    return regrets


def run_experiments_online(
    env: SLCBEnvironment,
    policy_learner: PolicyLearner,
    T: int,
    replay_buffer_size: int = 100,
) -> List[float]:
    """
    Runs online evaluation by training a policy learner on the
    data collected by following the attached `exploration_module`.
    """
    # prepare agent
    agent = PearlAgent(
        policy_learner=policy_learner,
        replay_buffer=DiscreteContextualBanditReplayBuffer(replay_buffer_size),
    )

    regrets = online_evaluation(env, agent, num_steps=T, training_mode=True)

    return regrets


def run_experiments(
    env: SLCBEnvironment,
    T: int,
    num_of_experiments: int,
    policy_learner_dict: Dict[str, Any],
    exploration_module_dict: Dict[str, Any],
    save_results_path: str,
    dataset_name: str,
    run_offline: bool = False,
) -> None:
    """
    Run experiments for a given policy learner and exploration module config.
    Has option to run online model and offline model
    (when the data is sampled via the uniform policy).
    """

    regrets = {}
    for experiment_num in range(num_of_experiments):
        print(
            "Running {} with exploration module {}. Experiment: {}".format(
                policy_learner_dict["name"],
                exploration_module_dict["name"],
                experiment_num,
            )
        )
        if not run_offline:
            # Online CB algorithms
            exploration_module = exploration_module_dict["method"](
                **exploration_module_dict["params"]
            )
            policy_learner_dict["params"]["exploration_module"] = exploration_module
            policy_learner = policy_learner_dict["method"](
                **policy_learner_dict["params"]
            )
            # override action representation module
            policy_learner._action_representation_module = policy_learner_dict[
                "params"
            ]["action_representation_module"]

            regret_single_run = run_experiments_online(
                env, policy_learner, T, replay_buffer_size=T
            )
            experiment_name = "method_{}_exploration_{}_experiment_num_{}".format(
                policy_learner_dict["name"],
                exploration_module_dict["name"],
                experiment_num,
            )
        else:
            # Offline evaluation
            regret_single_run = run_experiments_offline(
                env,
                T=run_config["T"],
                training_rounds=run_config["training_rounds"],
                hidden_dims=policy_learner_dict["params"]["hidden_dim"],
                num_eval_steps=policy_learner_dict["params"]["num_eval_steps"],
                action_representation_module=policy_learner_dict["params"][
                    "action_representation_module"
                ],
            )
            experiment_name = "offline_evaluation_experiment_num_{}".format(
                experiment_num
            )
        regrets[experiment_name] = regret_single_run

    # create save_results_path folder if doesnt exist
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)

    # delete existing result if exists
    save_results_path_name = os.path.join(
        save_results_path,
        "method_{}_exploration_{}_dataset_name_{}".format(
            policy_learner_dict["name"], exploration_module_dict["name"], dataset_name
        ),
    )
    if os.path.exists(save_results_path_name):
        os.remove(save_results_path_name)

    # save results
    df_regrets = pd.DataFrame(regrets)
    with open(save_results_path_name, "w") as file:
        df_regrets.to_csv(file)


def run_cb_benchmarks(
    cb_algorithms_config: Dict[str, Any],
    test_environments_config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    """
    Run Contextual Bandit algorithms on environments.
    cb_algorithms_config: dictionary with config files of the CB algorithms.
    test_environments_config: dictionary with config files of the test environments.
    run_config: dictionary with config files of the run parameters.
    """

    # Download uci datasets if dont exist
    uci_data_path = "./utils/instantiations/environments/uci_datasets"
    if not os.path.exists(uci_data_path):
        os.makedirs(uci_data_path)
        download_uci_data(data_path=uci_data_path)

    # Path to save results
    save_results_path: str = "./utils/scripts/cb_benchmark/experiments_results"
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    # run all CB algorithms on all benchmarks
    for algorithm in cb_algorithms_config.keys():
        for dataset_name in test_environments_config.keys():
            env = SLCBEnvironment(**test_environments_config[dataset_name])
            policy_learner_dict, exploration_module_dict = cb_algorithms_config[
                algorithm
            ](env)

            run_experiments(
                env=env,
                T=run_config["T"] if dataset_name != "letter" else 30000,
                num_of_experiments=run_config["num_of_experiments"],
                policy_learner_dict=policy_learner_dict,
                exploration_module_dict=exploration_module_dict,
                save_results_path=save_results_path,
                dataset_name=dataset_name,
                run_offline=algorithm == "OfflineEval",
            )


if __name__ == "__main__":

    # load CB algorithm
    cb_algorithms_config: Dict[str, Any] = {
        "NeuralSquareCB": return_neural_squarecb_config,
        "NeuralLinUCB": return_neural_lin_ucb_config,
        "NeuralLinTS": return_neural_lin_ts_config,
        "OfflineEval": return_offline_eval_config,
    }

    # load UCI dataset
    test_environments_config: Dict[str, Any] = {
        "pendigits": pendigits_uci_dict,
        "yeast": yeast_uci_dict,
        "letter": letter_uci_dict,
        "satimage": satimage_uci_dict,
    }

    run_cb_benchmarks(
        cb_algorithms_config=cb_algorithms_config,
        test_environments_config=test_environments_config,
        run_config=run_config,
    )
