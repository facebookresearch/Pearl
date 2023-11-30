#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import warnings
from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, Iterable, List

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.pearl_agent import PearlAgent
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

from pearl.user_envs.wrappers.gym_avg_torque_cost import GymAvgTorqueWrapper
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from torch import nn

warnings.filterwarnings("ignore")

number_of_episodes = 2000
save_path = "../fbsource/fbcode/pearl/"


class Evaluation(ABC):
    """
    Evaluation of an RL method on a given gym environment.
    Args:
        gym_environment_name: name of the gym environment to be evaluated
        *args: arguments passed to the constructor of the gym environment
        **kwargs: keyword arguments passed to the constructor of the gym environment
    """

    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        self.gym_environment_name: str = gym_environment_name
        self.device_id: int = device_id
        # pyre-fixme[4]: Attribute must be annotated.
        self.args = args
        # pyre-fixme[4]: Attribute must be annotated.
        self.kwargs = kwargs

    @abstractmethod
    def evaluate(self, seed: int = 0) -> Iterable[Number]:
        """Runs evaluation and returns sequence of obtained returns during training"""
        pass


class PearlDQN(Evaluation):
    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(PearlDQN, self).__init__(gym_environment_name, device_id, *args, **kwargs)

    def evaluate(self, seed: int) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name, *self.args, **self.kwargs)
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
            device_id=self.device_id,
        )
        info = online_learning(
            agent,
            env,
            number_of_episodes=number_of_episodes,
            learn_after_episode=True,
            print_every_x_episodes=1,
        )
        return info["return"]


class PearlLSTMDQN(Evaluation):
    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(PearlDQN, self).__init__(gym_environment_name, device_id, *args, **kwargs)

    def evaluate(self, seed: int) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name, *self.args, **self.kwargs)
        hidden_dim = 8
        action_space = env.action_space
        assert isinstance(action_space, DiscreteActionSpace)
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_actions=action_space.n
        )
        history_summarization_module = LSTMHistorySummarizationModule(
            observation_dim=env.observation_space.shape[0],
            action_dim=action_space.n,
            hidden_dim=hidden_dim,
        )
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=hidden_dim,
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
            ),
            action_representation_module=action_representation_module,
            history_summarization_module=history_summarization_module,
            replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
            device_id=self.device_id,
        )
        info = online_learning(
            agent,
            env,
            number_of_episodes=number_of_episodes,
            learn_after_episode=True,
            print_every_x_episodes=1,
        )
        return info["return"]


class PearlContinuousSAC(Evaluation):
    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(PearlContinuousSAC, self).__init__(
            gym_environment_name, device_id, *args, **kwargs
        )

    def evaluate(self, seed: int) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name, *self.args, **self.kwargs)
        agent = PearlAgent(
            policy_learner=ContinuousSoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[256, 256],
                critic_hidden_dims=[256, 256],
                training_rounds=1,
                batch_size=256,
                entropy_coef=0.05,
                entropy_autotune=True,
                actor_learning_rate=0.0003,
                critic_learning_rate=0.0005,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(100000),
            device_id=self.device_id,
        )
        info = online_learning(
            agent,
            env,
            number_of_episodes=number_of_episodes,
            learn_after_episode=False,
            print_every_x_episodes=1,
        )
        return info["return"]


class PearlPPO(Evaluation):
    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(PearlPPO, self).__init__(gym_environment_name, device_id, *args, **kwargs)

    def evaluate(self, seed: int) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name, *self.args, **self.kwargs)
        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[64, 64],
                critic_hidden_dims=[64, 64],
                training_rounds=50,
                batch_size=64,
                epsilon=0.1,
            ),
            replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
            device_id=self.device_id,
        )
        info = online_learning(
            agent,
            env,
            number_of_episodes=number_of_episodes,
            learn_after_episode=True,
            print_every_x_episodes=1,
        )
        return info["return"]


class PearlDDPG(Evaluation):
    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(PearlDDPG, self).__init__(
            gym_environment_name, device_id, *args, **kwargs
        )

    def evaluate(self, seed: int) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name, *self.args, **self.kwargs)
        agent = PearlAgent(
            policy_learner=DeepDeterministicPolicyGradient(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[256, 256],
                critic_hidden_dims=[256, 256],
                critic_learning_rate=3e-4,
                actor_learning_rate=3e-4,
                training_rounds=1,
                exploration_module=NormalDistributionExploration(
                    mean=0,
                    std_dev=0.1,
                ),
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(50000),
            device_id=self.device_id,
        )
        info = online_learning(
            agent,
            env,
            number_of_episodes=number_of_episodes,
            learn_after_episode=True,
            print_every_x_episodes=1,
        )
        return info["return"]


class PearlTD3(Evaluation):
    def __init__(
        self,
        gym_environment_name: str,
        device_id: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(PearlTD3, self).__init__(gym_environment_name, device_id, *args, **kwargs)

    def evaluate(self, seed: int) -> Iterable[Number]:
        has_cost_available = False
        if self.gym_environment_name[:3] == "wc_":
            has_cost_available = True
            self.gym_environment_name = self.gym_environment_name[3:]
            env = GymEnvironment(
                GymAvgTorqueWrapper(gym.make(self.gym_environment_name))
            )
        else:
            env = GymEnvironment(self.gym_environment_name, *self.args, **self.kwargs)
        agent = PearlAgent(
            policy_learner=TD3(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=[256, 256],
                critic_hidden_dims=[256, 256],
                critic_learning_rate=3e-4,
                actor_learning_rate=3e-4,
                training_rounds=1,
                exploration_module=NormalDistributionExploration(
                    mean=0,
                    std_dev=0.1,
                ),
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(50000),
            device_id=self.device_id,
        )
        # Enable saving cost in replay buffer if cost is available
        agent.replay_buffer.has_cost_available = has_cost_available

        info = online_learning(
            agent,
            env,
            number_of_episodes=number_of_episodes,
            learn_after_episode=True,
            print_every_x_episodes=1,
        )
        return info["return"]


def evaluate(evaluations: Iterable[Evaluation]) -> None:
    """Obtain data from evaluations and plot them, one plot per environment"""
    num_seeds = 5
    for seed in range(num_seeds):
        set_seed(seed)  # seed all sources of randomness except the envronment reset
        print(f"Seed {seed}")
        data_by_environment_and_method = collect_data(evaluations, seed=seed)
        generate_plots(data_by_environment_and_method, seed=seed)


def collect_data(
    evaluations: Iterable[Evaluation], seed: int
) -> Dict[str, Dict[str, List[float]]]:
    data_by_environment_and_method = {}
    for evaluation in evaluations:
        method = type(evaluation).__name__
        environment = evaluation.gym_environment_name
        print(f"Running {method} on {environment} ...")
        returns = evaluation.evaluate(seed=seed)  # to set the environment seed
        if environment not in data_by_environment_and_method:
            data_by_environment_and_method[environment] = {}
        data_by_environment_and_method[environment][method] = returns
        dir_name = save_path + str(method) + "/" + str(environment) + "/"
        os.makedirs(dir_name, exist_ok=True)
        with open(
            dir_name + "returns_data_seed_" + str(seed) + ".pickle", "wb"
        ) as handle:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(
                data_by_environment_and_method, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
    return data_by_environment_and_method


def generate_plots(
    data_by_environment_and_method: Dict[str, Dict[str, List[float]]],
    seed: int,
) -> None:
    for environment_name, data_by_method in data_by_environment_and_method.items():
        plt.title(environment_name)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        method, _ = next(iter(data_by_method.items()))
        window_size = 10

        for method, returns in data_by_method.items():
            plt.plot(returns, label=method)
            rolling_mean_returns = (
                np.convolve(returns, np.ones(window_size), "valid") / window_size
            )

            plt.plot(rolling_mean_returns, label=f"Rolling Mean {method}")
        plt.legend()
        dir_name = save_path + str(method) + "/" + str(environment_name) + "/"
        os.makedirs(dir_name, exist_ok=True)
        filename = f"{environment_name} and {method} and seed = {seed}.png"
        logging.info(f"Saving plot to {os.getcwd()}/{filename}")
        plt.savefig(filename)
        plt.savefig(dir_name + "/" + filename)
        plt.close()


def tianshou_dqn_cart_pole() -> int:
    """Evaluates a set of algorithms on a set of environments"""

    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(np.prod(state_shape), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape)),
            )

        def forward(self, obs, state=None, info=None):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # environments
    env = gym.make("CartPole-v1")
    # pyre-fixme[6]: For 1st argument expected `List[typing.Callable[[], Env]]` but
    #  got `List[typing.Callable[[], Env[typing.Any, typing.Any]]]`.
    train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
    # pyre-fixme[6]: For 1st argument expected `List[typing.Callable[[], Env]]` but
    #  got `List[typing.Callable[[], Env[typing.Any, typing.Any]]]`.
    test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])

    # model & optimizer
    # pyre-fixme[16]: `Space` has no attribute `n`.
    net = Net(env.observation_space.shape, env.action_space.n).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=0.0003)

    # DQN policy
    policy = DQNPolicy(
        net,
        optim,
        action_space=env.action_space,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=320,
    )

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(total_size=20000, buffer_num=20),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=10000,
        step_per_collect=10,
        update_per_step=0.1,
        episode_per_test=100,
        batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    )
    print(result)
    return 0


def tianshou_ppo_cart_pole() -> int:
    """Evaluates a set of algorithms on a set of environments"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # environments
    env = gym.make("CartPole-v1")
    # pyre-fixme[6]: For 1st argument expected `List[typing.Callable[[], Env]]` but
    #  got `List[typing.Callable[[], Env[typing.Any, typing.Any]]]`.
    train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
    # pyre-fixme[6]: For 1st argument expected `List[typing.Callable[[], Env]]` but
    #  got `List[typing.Callable[[], Env[typing.Any, typing.Any]]]`.
    test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])

    # model & optimizer
    # pyre-fixme[6]: For 1st argument expected `Union[Sequence[int], int]` but got
    #  `Optional[typing.Tuple[int, ...]]`.
    net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
    # pyre-fixme[16]: `Space` has no attribute `n`.
    actor = Actor(net, env.action_space.n, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    # PPO policy
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        action_space=env.action_space,
        deterministic_eval=True,
    )

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(20000, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= env.spec.reward_threshold,
    )
    print(result)
    return 0


def main(device_id: int = -1) -> None:
    # TODO: this should be part of argparse instead of hardcoded.
    evaluate(
        [
            # Methods applied to the same environment will be grouped in the same plot.
            # All receive device id
            # PearlDQN("CartPole-v1", device_id=device_id),
            # PearlPPO("CartPole-v1", device_id=device_id),
            # PearlDQN("Acrobot-v1", device_id=device_id),
            # PearlPPO("Acrobot-v1", device_id=device_id),
            # PearlDDPG("Pendulum-v1", device_id=device_id),
            PearlTD3("wc_Pendulum-v1", device_id=device_id),
            # MuJoCo environments -- require MuJoCo to be installed.
            # PearlDDPG("HalfCheetah-v4"),
            # PearlDDPG("Ant-v4"),
            # PearlDDPG("Hopper-v4")
            # PearlDDPG("Walker2d-v4")
            # PearlTD3("HalfCheetah-v4"),
            # PearlTD3("Ant-v4"),
            # PearlTD3("Hopper-v4"),
            # PearlTD3("Walker2d-v4"),
            # PearlContinuousSAC("Ant-v4")
        ]
    )

    # tianshou_dqn_cart_pole()
    # tianshou_ppo_cart_pole()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pearl Benchmark")
    parser.add_argument(
        "-d",
        "--device",
        help="GPU device ID (optional)",
        required=False,
        type=int,
        default=-1,
    )
    args: argparse.Namespace = parser.parse_args()

    if args.device != -1:
        print(f"Going to attempt to use GPU (cuda:{args.device})")
    else:
        print("Going to attempt to use CPU")

    main(args.device)
