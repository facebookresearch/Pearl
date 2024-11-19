# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch

from pearl.policy_learners.sequential_decision_making.ppo import (
    PPOReplayBuffer,
    PPOTransition,
    ProximalPolicyOptimization,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestPPO(unittest.TestCase):
    def test_optimizer_param_count(self) -> None:
        """
        This test is to ensure optimizer defined in PPO has all the parameters needed
        including actor and critic
        """
        policy_learner = ProximalPolicyOptimization(
            16,
            DiscreteActionSpace(actions=[torch.tensor(i) for i in range(3)]),
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            training_rounds=1,
            batch_size=500,
            epsilon=0.1,
        )
        optimizer_params_count = sum(
            len(group["params"])
            for group in policy_learner._actor_optimizer.param_groups
            + policy_learner._critic_optimizer.param_groups
        )
        model_params_count = sum([1 for _ in policy_learner._actor.parameters()]) + sum(
            [1 for _ in policy_learner._critic.parameters()]
        )
        self.assertEqual(optimizer_params_count, model_params_count)

    def test_preprocess_replay_buffer(self) -> None:
        """
        PPO computes generalized advantage estimation and truncated lambda return
        This test is to ensure the calculation is correct.
        """
        state_dim = 1
        action_space = DiscreteActionSpace(actions=[torch.tensor(i) for i in range(3)])
        policy_learner = ProximalPolicyOptimization(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            training_rounds=10,
            batch_size=500,
            epsilon=0.1,
            discount_factor=0.6,
            trace_decay_param=0.5,
        )
        capacity = 10
        rewards = [4.0, 6.0, 5.0]
        trajectory_len = len(rewards)
        replay_buffer = PPOReplayBuffer(capacity)
        for i in range(trajectory_len):
            replay_buffer.push(
                state=torch.tensor([i * 1.0]),
                action=torch.tensor(i * 1.0),
                reward=rewards[i],
                next_state=torch.tensor([i * 1.0]),
                curr_available_actions=action_space,
                next_available_actions=action_space,
                terminated=False,
                max_number_actions=action_space.n,
            )
        # gaes:
        # gae0 = 4 + 0.6 * v1 - v0 + 0.6 * 0.5 * gae1 --> state 0
        # gae1 = 6 + 0.6 * v2 - v1 + 0.6 * 0.5 * gae2 --> state 1
        # gae2 = 5 + 0.6 * v3 - v2 --> state 2
        # truncated lambda returns:
        # lam_return0 = gae0 + v0
        # lam_return1 = gae1 + v1
        # lam_return2 = gae2 + v2

        v0 = policy_learner._critic(replay_buffer.memory[0].state).detach()[0]
        v1 = policy_learner._critic(replay_buffer.memory[1].state).detach()[0]
        v2 = policy_learner._critic(replay_buffer.memory[2].state).detach()[0]
        v3 = policy_learner._critic(replay_buffer.memory[2].next_state).detach()[0]
        gae2 = 5 + 0.6 * v3 - v2
        gae1 = 6 + 0.6 * v2 - v1 + 0.6 * 0.5 * gae2
        gae0 = 4 + 0.6 * v1 - v0 + 0.6 * 0.5 * gae1
        lam_return2 = gae2 + v2
        lam_return1 = gae1 + v1
        lam_return0 = gae0 + v0
        true_gaes = [gae0, gae1, gae2]
        true_lambda_returns = [lam_return0, lam_return1, lam_return2]  # list of returns

        policy_learner.preprocess_replay_buffer(replay_buffer)
        for i in range(trajectory_len):
            transition = replay_buffer.memory[i]

            assert isinstance(transition, PPOTransition)
            gae = transition.gae
            assert gae is not None
            self.assertEqual(true_gaes[i], gae.detach())

            lam_return = transition.lam_return
            assert lam_return is not None
            self.assertEqual(true_lambda_returns[i], lam_return.detach())
