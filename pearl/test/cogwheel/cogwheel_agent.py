#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Cogwheel wiki
https://www.internalfb.com/intern/wiki/Cogwheel/

To RUN:
    buck2 run //pearl:cogwheel_pearl_agent-launcher

This takes a long time (usually 3 hours on a devserver), which is why it is on cogwheel where it will not time out.
"""
import logging

from pearl.core.common.neural_networks.value_networks import (
    DuelingStateActionValueNetwork,
)

from pearl.core.common.pearl_agent import PearlAgent
from pearl.core.common.replay_buffer.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.core.common.replay_buffer.fifo_on_policy_replay_buffer import (
    FIFOOnPolicyReplayBuffer,
)
from pearl.core.common.replay_buffer.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
)

from pearl.core.sequential_decision_making.policy_learners.deep_q_learning import (
    DeepQLearning,
)

from pearl.core.sequential_decision_making.policy_learners.deep_sarsa import DeepSARSA
from pearl.core.sequential_decision_making.policy_learners.policy_gradient import (
    PolicyGradient,
)
from pearl.core.sequential_decision_making.policy_learners.ppo import (
    ProximalPolicyOptimization,
)
from pearl.core.sequential_decision_making.policy_learners.soft_actor_critic import (
    SoftActorCritic,
)

from pearl.gym.gym_environment import GymEnvironment
from pearl.online_learning.online_learning import target_return_is_reached
from windtunnel.cogwheel.test import cogwheel_test, CogwheelTest


class TestAgent(CogwheelTest):
    @cogwheel_test
    def test_dqn(self) -> None:
        """
        This test is checking if DQN will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
        )
        self.assertTrue(
            target_return_is_reached(
                target_return=500,
                max_episodes=1000,
                agent=agent,
                env=env,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )

    @cogwheel_test
    def test_double_dqn(self) -> None:
        """
        This test is checking if double DQN will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                training_rounds=20,
                double=True,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
        )
        self.assertTrue(
            target_return_is_reached(
                target_return=500,
                max_episodes=1000,
                agent=agent,
                env=env,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )

    @cogwheel_test
    def test_sarsa(self) -> None:
        """
        This test is checking if SARSA will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepSARSA(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                training_rounds=20,
            ),
            replay_buffer=FIFOOnPolicyReplayBuffer(10_000),
        )
        self.assertTrue(
            target_return_is_reached(
                target_return=500,
                max_episodes=1000,
                agent=agent,
                env=env,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )

    @cogwheel_test
    def test_pg(self) -> None:
        """
        This test is checking if Policy Gradient will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=PolicyGradient(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                batch_size=500,
            ),
            replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
        )
        self.assertTrue(
            target_return_is_reached(
                target_return=500,
                max_episodes=10_000,
                agent=agent,
                env=env,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )

    @cogwheel_test
    def test_dueling_dqn(
        self,
        batch_size: int = 128,
    ) -> None:
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                env.observation_space.shape[0],
                env.action_space,
                hidden_dims=[64],
                training_rounds=20,
                network_type=DuelingStateActionValueNetwork,
                batch_size=batch_size,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
        )
        self.assertTrue(
            target_return_is_reached(
                agent=agent,
                env=env,
                target_return=500,
                max_episodes=10_000,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )

    @cogwheel_test
    def test_ppo(self) -> None:
        """
        This test is checking if PPO using cumulated returns will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                training_rounds=50,
                batch_size=64,
                epsilon=0.1,
            ),
            replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
        )
        self.assertTrue(
            target_return_is_reached(
                agent=agent,
                env=env,
                target_return=500,
                max_episodes=10_000,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )

    @cogwheel_test
    def test_sac(self) -> None:
        """
        This test is checking if SAC will eventually get to 500 return for CartPole-v1
        """
        env = GymEnvironment("CartPole-v1")
        agent = PearlAgent(
            policy_learner=SoftActorCritic(
                env.observation_space.shape[0],
                env.action_space,
                [64, 64],
                training_rounds=100,
                batch_size=100,
                entropy_coef=0.1,
                learning_rate=0.0003,
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(50000),
        )
        self.assertTrue(
            target_return_is_reached(
                agent=agent,
                env=env,
                target_return=500,
                max_episodes=10_000,
                learn=True,
                learn_after_episode=True,
                exploit=False,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    TestAgent().main()
