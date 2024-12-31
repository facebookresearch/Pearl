# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, List, Optional

import torch
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule
from pearl.utils.functional_utils.learning.critic_utils import (
    make_critic,
    twin_critic_action_value_loss,
    update_critic_target_network,
)
from pearl.utils.module_utils import modules_have_similar_state_dict
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
        critic_hidden_dims: list[int],
        lambda_constraint_ub_value: float = 20.0,
        lambda_constraint_init_value: float = 0.0,
        cost_discount_factor: float = 0.5,
        lr_lambda: float = 1e-2,
        critic_learning_rate: float = 1e-3,
        critic_soft_update_tau: float = 0.005,
        batch_size: int = 256,
        use_twin_critic: bool = True,
        critic_network_type: type[QValueNetwork] = VanillaQValueNetwork,
        cost_critic_optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        super().__init__()

        # initialize parameters of lambda constraint
        self.constraint_value = constraint_value
        self.lr_lambda = lr_lambda
        self.lambda_constraint_ub_value = lambda_constraint_ub_value
        self.lambda_constraint = lambda_constraint_init_value
        self._batch_size = batch_size

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
        if cost_critic_optimizer is not None:
            self.cost_critic_optimizer: optim.Optimizer = cost_critic_optimizer
        else:
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
            1,
        )

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        if len(replay_buffer) == 0:
            return

        if self._batch_size == -1 or len(replay_buffer) < self._batch_size:
            batch_size = len(replay_buffer)
        else:
            batch_size = self._batch_size

        batch = replay_buffer.sample(batch_size)
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
        Update the lambda constraint based on the cost critic via a projected gradient descent
        update rule.
        """

        with torch.no_grad():
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            cost_q1, cost_q2 = cost_critic.get_q_values(
                state_batch=batch.state,
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `sample_action`.
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
    ) -> dict[str, Any]:
        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sample_action`.
            next_action = policy_learner._actor.sample_action(batch.next_state)

            # sample q values of (next_state, next_action) from targets of critics
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
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
                next_q * self.cost_discount_factor * (1 - batch.terminated.float())
                # pyre-fixme[58]: `+` is not supported for operand types `Tensor` and
                #  `Optional[Tensor]`.
            ) + batch.cost  # (batch_size)

        # update twin critics towards bellman target
        assert isinstance(self.cost_critic, TwinCritic)
        loss, _, _ = twin_critic_action_value_loss(
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

    def compare(self, other: SafetyModule) -> str:
        """
        Compares two RCSafetyModuleCostCriticContinuousAction instances for equality,
        checking attributes and the cost critic.

        Args:
          other: The other SafetyModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, RCSafetyModuleCostCriticContinuousAction):
            differences.append(
                "other is not an instance of RCSafetyModuleCostCriticContinuousAction"
            )
        else:  # Type refinement with else block
            # Compare attributes
            if self.constraint_value != other.constraint_value:
                differences.append(
                    f"constraint_value is different: {self.constraint_value} "
                    + f"vs {other.constraint_value}"
                )
            if self.lr_lambda != other.lr_lambda:
                differences.append(
                    f"lr_lambda is different: {self.lr_lambda} "
                    + f"vs {other.lr_lambda}"
                )
            if self.lambda_constraint_ub_value != other.lambda_constraint_ub_value:
                differences.append(
                    f"lambda_constraint_ub_value is different: {self.lambda_constraint_ub_value} "
                    + f"vs {other.lambda_constraint_ub_value}"
                )
            if self.lambda_constraint != other.lambda_constraint:
                differences.append(
                    f"lambda_constraint is different: {self.lambda_constraint} "
                    + f"vs {other.lambda_constraint}"
                )
            if self._batch_size != other._batch_size:
                differences.append(
                    f"_batch_size is different: {self._batch_size} "
                    + f"vs {other._batch_size}"
                )
            if self.use_twin_critic != other.use_twin_critic:
                differences.append(
                    f"use_twin_critic is different: {self.use_twin_critic} "
                    + f"vs {other.use_twin_critic}"
                )
            if self.state_dim != other.state_dim:
                differences.append(
                    f"state_dim is different: {self.state_dim} "
                    + f"vs {other.state_dim}"
                )
            if self.action_dim != other.action_dim:
                differences.append(
                    f"action_dim is different: {self.action_dim} "
                    + f"vs {other.action_dim}"
                )
            if self.hidden_dims != other.hidden_dims:
                differences.append(
                    f"hidden_dims is different: {self.hidden_dims} "
                    + f"vs {other.hidden_dims}"
                )
            if self.critic_learning_rate != other.critic_learning_rate:
                differences.append(
                    f"critic_learning_rate is different: {self.critic_learning_rate} "
                    + f"vs {other.critic_learning_rate}"
                )
            if self.cost_discount_factor != other.cost_discount_factor:
                differences.append(
                    f"cost_discount_factor is different: {self.cost_discount_factor} "
                    + f"vs {other.cost_discount_factor}"
                )
            if self.critic_soft_update_tau != other.critic_soft_update_tau:
                differences.append(
                    f"critic_soft_update_tau is different: {self.critic_soft_update_tau} "
                    + f"vs {other.critic_soft_update_tau}"
                )

            # Compare cost critics using modules_have_similar_state_dict
            if (
                reason := modules_have_similar_state_dict(
                    self.cost_critic, other.cost_critic
                )
            ) != "":
                differences.append(f"cost_critic is different: {reason}")

        return "\n".join(differences)
