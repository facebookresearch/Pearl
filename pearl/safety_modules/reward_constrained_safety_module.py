# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, List, Type

import torch
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    make_critic,
    twin_critic_action_value_loss,
    update_critic_target_network,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule
from torch import nn, optim


class RCSafetyModuleCostCriticContinuousAction(SafetyModule):
    """
    Safety reward model with learned cost critic for continous actions setting.
    The critic learns the cost q function of the policy.
    The critic is later being used to update the lagrange multiplier,
    constraint_value, being used in policy_learner in pre-processing of batch.

    """

    def __init__(
        self,
        constraint_value: float,
        state_dim: int,
        action_space: ActionSpace,
        critic_hidden_dims: List[int],
        lambda_constraint_ub_value: float = 20.0,
        lambda_constraint_init_value: float = 0.0,
        cost_discount_factor: float = 0.5,
        lr_lambda: float = 1e-2,
        critic_learning_rate: float = 1e-3,
        critic_soft_update_tau: float = 0.005,
        batch_size: int = 256,
        use_twin_critic: bool = True,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
    ) -> None:
        super(RCSafetyModuleCostCriticContinuousAction, self).__init__()

        # initialize parameters of lambda constraint
        self.constraint_value = constraint_value
        self.lr_lambda = lr_lambda
        self.lambda_constraint_ub_value = lambda_constraint_ub_value
        self.lambda_constraint = lambda_constraint_init_value
        self.batch_size = batch_size

        # initialize parameters of cost critic
        assert hasattr(action_space, "action_dim")
        self.action_space = action_space
        self.use_twin_critic = use_twin_critic
        self.state_dim = state_dim
        self.action_dim: int = self.action_space.action_dim
        self.hidden_dims = critic_hidden_dims
        self.critic_learning_rate = critic_learning_rate
        self.cost_discount_factor = cost_discount_factor
        self.critic_soft_update_tau = critic_soft_update_tau

        # initialize cost critic and target cost critic
        self.cost_critic: nn.Module = make_critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            use_twin_critic=self.use_twin_critic,
            network_type=critic_network_type,
        )
        self.cost_critic_optimizer = optim.AdamW(
            [
                {
                    "params": self.cost_critic.parameters(),
                    "lr": self.critic_learning_rate,
                    "amsgrad": True,
                },
            ]
        )
        self.target_of_cost_critic: nn.Module = make_critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            use_twin_critic=self.use_twin_critic,
            network_type=critic_network_type,
        )
        update_critic_target_network(
            self.target_of_cost_critic,
            self.cost_critic,
            self.use_twin_critic,
            1,
        )

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        if len(replay_buffer) < self.batch_size or len(replay_buffer) == 0:
            return

        batch = replay_buffer.sample(self.batch_size)
        assert isinstance(batch, TransitionBatch)
        batch = policy_learner.preprocess_batch(batch)

        # train critic
        self.cost_critic_learn_batch(batch, policy_learner)

        # update the lambda value via soft update
        self.constraint_lambda_update(batch, policy_learner, self.cost_critic)

    def constraint_lambda_update(
        self,
        batch: TransitionBatch,
        policy_learner: PolicyLearner,
        cost_critic: nn.Module,
    ) -> None:

        """
        Update the lambda constraint based on the cost critic via a projected gradient descent update rule.
        """

        with torch.no_grad():
            cost_q1, cost_q2 = cost_critic.get_q_values(
                state_batch=batch.state,
                action_batch=policy_learner._actor.sample_action(batch.state),
            )
            cost_q = torch.maximum(cost_q1, cost_q2)
            cost_q = cost_q.mean().item()

        # projected gradient descent step update
        lambda_update = self.lambda_constraint + self.lr_lambda * (
            cost_q * (1 - self.cost_discount_factor) - self.constraint_value
        )
        lambda_update = max(lambda_update, 0.0)
        lambda_update = min(lambda_update, self.lambda_constraint_ub_value)
        self.lambda_constraint = lambda_update

    def cost_critic_learn_batch(
        self, batch: TransitionBatch, policy_learner: PolicyLearner
    ) -> Dict[str, Any]:

        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            next_action = policy_learner._actor.sample_action(batch.next_state)

            # sample q values of (next_state, next_action) from targets of critics
            next_q1, next_q2 = self.target_of_cost_critic.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # cost + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self.cost_discount_factor * (1 - batch.done.float())
            ) + batch.cost  # (batch_size)

        # update twin critics towards bellman target
        assert isinstance(self.cost_critic, TwinCritic)
        loss = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self.cost_critic,
        )
        self.cost_critic_optimizer.zero_grad()
        loss.backward()
        self.cost_critic_optimizer.step()
        # update targets of critics using soft updates
        update_critic_target_network(
            self.target_of_cost_critic,
            self.cost_critic,
            self.use_twin_critic,
            self.critic_soft_update_tau,
        )

        return {
            "cost_critic_loss": loss.item(),
        }

    def filter_action(
        self, subjective_state: SubjectiveState, action_space: ActionSpace
    ) -> ActionSpace:
        return action_space

    def learn_batch(
        self, batch: TransitionBatch, policy_learner: PolicyLearner
    ) -> None:
        """
        Don't support offline training for now.
        """
        AssertionError("Offline safety learning is not supported yet.")
        pass

    def __str__(self) -> str:
        return "RCSafetyModuleCostCriticContinuousAction"
