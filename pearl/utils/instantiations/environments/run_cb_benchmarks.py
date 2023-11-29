# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import random
from typing import Any, Dict, List

import pandas as pd

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.policy_learner import PolicyLearner

from pearl.replay_buffers.contextual_bandits.discrete_contextual_bandit_replay_buffer import (
    DiscreteContextualBanditReplayBuffer,
)
from pearl.utils.instantiations.environments.cb_benchmark_config import (
    letter_uci_dict,
    pendigits_uci_dict,
    return_NeuralFastCBConfig,
    return_NeuralLinTSConfig,
    return_NeuralLinUCBConfig,
    return_NeuralSquareCBConfig,
    return_offlineEvalConfig,
    run_config,
    satimage_uci_dict,
    yeast_uci_dict,
)
from pearl.utils.instantiations.environments.contextual_bandit_linear_synthetic_environment import (
    SLCBEnvironment,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


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
    action_embeddings: str = "binary_embedding",
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
    hidden_dims: List[int] = [64, 16],
    num_eval_steps: int = 5000,
) -> List[float]:
    """
    Runs offline evaluation by training a NeuralBandit on the data collected by taking uniform actions
    """
    feature_dim = env.observation_dim
    dim_actions = env._action_dim_env

    # prepare offline agent
    neural_greedy_policy = NeuralBandit(
        feature_dim=feature_dim + dim_actions,
        hidden_dims=hidden_dims,
        learning_rate=0.01,
        batch_size=128,
        training_rounds=T,
        exploration_module=NoExploration(),
        use_keyed_optimizer=False,
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
    Runs online evaluation by training a policy learner on the data collected by following exploration_module
    """
    # prepare agent
    agent = PearlAgent(
        policy_learner=policy_learner,
        replay_buffer=DiscreteContextualBanditReplayBuffer(replay_buffer_size),
        action_representation_module=policy_learner._action_representation_module,
    )

    regrets = online_evaluation(env, agent, num_steps=T, training_mode=True)

    return regrets


def write_to_manifold(manifold_path_name: str, df: pd.DataFrame) -> None:
    pathmgr = PathManager()
    pathmgr.register_handler(ManifoldPathHandler(timeout_sec=180), allow_override=True)
    with pathmgr.open(manifold_path_name, "w") as file:
        df.to_csv(file)
    return


def run_experiments(
    env: SLCBEnvironment,
    T: int,
    num_of_experiments: int,
    policy_learner_dict: Dict[str, Any],
    exploration_module_dict: Dict[str, Any],
    manifold_path: str,
    dataset_name: str,
    run_offline: bool = False,
) -> None:
    """
    Run experiments for a given policy learner and exploration module config.
    Has option to run online model and offline model (when the data is sampled via the uniform policy).
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
            )
            experiment_name = "offline_evaluation_experiment_num_{}".format(
                experiment_num
            )
        regrets[experiment_name] = regret_single_run

    # write to Manifold
    df_regrets = pd.DataFrame(regrets)
    manifold_path_name = os.path.join(
        manifold_path,
        "method_{}_exploration_{}_dataset_name_{}".format(
            policy_learner_dict["name"], exploration_module_dict["name"], dataset_name
        ),
    )
    write_to_manifold(manifold_path_name, df_regrets)


if __name__ == "__main__":
    MANIFOLD_PATH: str = "manifold://cb_datasets/tree/uci_datasets"
    save_experiments_path: str = os.path.join(MANIFOLD_PATH, "experiments")

    # load UCI dataset
    valid_env_dict: Dict[str, Any] = {
        "pendigits": pendigits_uci_dict,
        "yeast": yeast_uci_dict,
        "letter": letter_uci_dict,
        "satimage": satimage_uci_dict,
    }
    dataset_name = "satimage"
    env = SLCBEnvironment(**valid_env_dict[dataset_name])
    policy_learner_dict: Dict[str, Any] = {}
    exploration_module_dict: Dict[str, Any] = {}

    # load CB algorithm
    return_CB_config: Dict[str, Any] = {
        "NeuralSquareCB": return_NeuralSquareCBConfig,
        "NeuralFastCB": return_NeuralFastCBConfig,
        "NeuralLinTS": return_NeuralLinTSConfig,
        "NeuralLinUCB": return_NeuralLinUCBConfig,
        "OfflineEval": return_offlineEvalConfig,
    }

    # Run offline evaluation
    algorithm = "OfflineEval"
    policy_learner_dict, exploration_module_dict = return_CB_config[algorithm](env)

    run_experiments(
        env=env,
        T=run_config["T"],
        num_of_experiments=run_config["num_of_experiments"],
        policy_learner_dict=policy_learner_dict,
        exploration_module_dict=exploration_module_dict,
        manifold_path=save_experiments_path,
        dataset_name=dataset_name,
        run_offline=algorithm == "OfflineEval",
    )

    # Run NeuralSqaureCB
    algorithm = "NeuralSquareCB"
    policy_learner_dict, exploration_module_dict = return_CB_config[algorithm](env)

    run_experiments(
        env=env,
        T=run_config["T"],
        num_of_experiments=run_config["num_of_experiments"],
        policy_learner_dict=policy_learner_dict,
        exploration_module_dict=exploration_module_dict,
        manifold_path=save_experiments_path,
        dataset_name=dataset_name,
        run_offline=algorithm == "OfflineEval",
    )

    # Run NeuralFastCB
    algorithm = "NeuralFastCB"
    policy_learner_dict, exploration_module_dict = return_CB_config[algorithm](env)

    run_experiments(
        env=env,
        T=run_config["T"],
        num_of_experiments=run_config["num_of_experiments"],
        policy_learner_dict=policy_learner_dict,
        exploration_module_dict=exploration_module_dict,
        manifold_path=save_experiments_path,
        dataset_name=dataset_name,
        run_offline=algorithm == "OfflineEval",
    )

    # Run NeuralLinUCB
    algorithm = "NeuralLinUCB"
    policy_learner_dict, exploration_module_dict = return_CB_config[algorithm](env)

    run_experiments(
        env=env,
        T=run_config["T"],
        num_of_experiments=run_config["num_of_experiments"],
        policy_learner_dict=policy_learner_dict,
        exploration_module_dict=exploration_module_dict,
        manifold_path=save_experiments_path,
        dataset_name=dataset_name,
        run_offline=algorithm == "OfflineEval",
    )

    # Run NeuralLinTS
    algorithm = "NeuralLinTS"
    policy_learner_dict, exploration_module_dict = return_CB_config[algorithm](env)

    run_experiments(
        env=env,
        T=run_config["T"],
        num_of_experiments=run_config["num_of_experiments"],
        policy_learner_dict=policy_learner_dict,
        exploration_module_dict=exploration_module_dict,
        manifold_path=save_experiments_path,
        dataset_name=dataset_name,
        run_offline=algorithm == "OfflineEval",
    )
