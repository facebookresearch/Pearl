#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import warnings
from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, Iterable

import gym

import matplotlib.pyplot as plt
import numpy as np
import torch
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (
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
from pearl.policy_learners.sequential_decision_making.td3 import TD3
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.sequential_decision_making.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import (
    online_learning_returns,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from torch import nn

warnings.filterwarnings("ignore")

number_of_episodes = 300


class Evaluation(ABC):
    """Evaluation of an RL method on a given environment."""

    def __init__(self, gym_environment_name):
        self.gym_environment_name = gym_environment_name

    @abstractmethod
    def evaluate(self) -> Iterable[Number]:
        """Runs evaluation and returns sequence of obtained returns during training"""
        pass


class PearlDQN(Evaluation):
    def __init__(self, gym_environment_name):
        super(PearlDQN, self).__init__(gym_environment_name)

    def evaluate(self) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name)
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
        )
        returns = online_learning_returns(
            agent, env, number_of_episodes=number_of_episodes, learn_after_episode=True
        )
        return returns


class PearlPPO(Evaluation):
    def __init__(self, gym_environment_name):
        super(PearlPPO, self).__init__(gym_environment_name)

    def evaluate(self) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name)
        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=50,
                batch_size=64,
                epsilon=0.1,
            ),
            replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
        )
        returns = online_learning_returns(
            agent, env, number_of_episodes=number_of_episodes, learn_after_episode=True
        )
        return returns


class PearlDDPG(Evaluation):
    def __init__(self, gym_environment_name):
        super(PearlDDPG, self).__init__(gym_environment_name)

    def evaluate(self) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name)
        agent = PearlAgent(
            policy_learner=DeepDeterministicPolicyGradient(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                hidden_dims=[400, 300],
                exploration_module=NormalDistributionExploration(
                    mean=0, std_dev=0.2, max_action_value=2, min_action_value=-2
                ),
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(50000),
        )
        returns = online_learning_returns(
            agent, env, number_of_episodes=number_of_episodes, learn_after_episode=True
        )
        return returns


class PearlTD3(Evaluation):
    def __init__(self, gym_environment_name):
        super(PearlTD3, self).__init__(gym_environment_name)

    def evaluate(self) -> Iterable[Number]:
        env = GymEnvironment(self.gym_environment_name)
        agent = PearlAgent(
            policy_learner=TD3(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                hidden_dims=[400, 300],
                exploration_module=NormalDistributionExploration(
                    mean=0, std_dev=0.2, max_action_value=2, min_action_value=-2
                ),
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(50000),
        )
        returns = online_learning_returns(
            agent, env, number_of_episodes=number_of_episodes, learn_after_episode=True
        )
        return returns


def evaluate(evaluations: Iterable[Evaluation]):
    """Obtain data from evaluations and plot them, one plot per environment"""
    data_by_environment_and_method = collect_data(evaluations)
    generate_plots(data_by_environment_and_method)


def collect_data(evaluations: Iterable[Evaluation]) -> Dict[str, str]:
    data_by_environment_and_method = {}
    for evaluation in evaluations:
        method = type(evaluation).__name__
        environment = evaluation.gym_environment_name
        print(f"Running {method} on {environment} ...")
        returns = evaluation.evaluate()
        if environment not in data_by_environment_and_method:
            data_by_environment_and_method[environment] = {}
        data_by_environment_and_method[environment][method] = returns
    return data_by_environment_and_method


def generate_plots(data_by_environment_and_method: Dict[str, str]):
    for environment_name, data_by_method in data_by_environment_and_method.items():
        plt.title(environment_name)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        for method, returns in data_by_method.items():
            plt.plot(returns, label=method)
            window_size = 5
            rolling_mean_returns = (
                np.convolve(returns, np.ones(window_size), "valid") / window_size
            )

            plt.plot(rolling_mean_returns, label=f"Rolling Mean {method}")
        plt.legend()
        plt.savefig(f"{environment_name}.png")
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
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # environments
    env = gym.make("CartPole-v1")
    train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
    test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])

    # model & optimizer
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
    train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
    test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])

    # model & optimizer
    net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
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


if __name__ == "__main__":
    evaluate(
        [
            # Methods applied to the same environment will be grouped in the same plot.
            PearlDQN("CartPole-v1"),
            PearlPPO("CartPole-v1"),
            PearlDQN("Acrobot-v1"),
            PearlPPO("Acrobot-v1"),
            PearlDDPG("Pendulum-v1"),
            PearlTD3("Pendulum-v1"),
        ]
    )

    # tianshou_dqn_cart_pole()
    # tianshou_ppo_cart_pole()
