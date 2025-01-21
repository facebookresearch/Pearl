# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace

from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (  # noqa E501
    NormalDistributionExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    twin_critic_action_value_loss,
)
from torch import nn, optim


class DeepDeterministicPolicyGradient(ActorCriticBase):
    """
    A Class for Deep Deterministic Deep Policy Gradient policy learner.
    paper: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(
        self,
        action_space: ActionSpace,
        state_dim: int | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        exploration_module: ExplorationModule | None = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        history_summarization_learning_rate: float = 1e-3,
        actor_network_type: type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        action_representation_module: ActionRepresentationModule | None = None,
        actor_network_instance: ActorNetwork | None = None,
        critic_network_instance: QValueNetwork | nn.Module | None = None,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
        history_summarization_optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            history_summarization_learning_rate=history_summarization_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=True,
            use_critic_target=True,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            use_twin_critic=True,  # we need to make this optional to users
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else NormalDistributionExploration(mean=0.0, std_dev=0.1)
            ),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=True,
            on_policy=False,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            history_summarization_optimizer=history_summarization_optimizer,
        )

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        # sample a batch of actions from the actor network; shape (batch_size, action_dim)
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        action_batch = self._actor.sample_action(batch.state)

        # obtain q values for (batch.state, action_batch) from critic 1
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `get_q_values`.
        q1 = self._critic._critic_1.get_q_values(
            state_batch=batch.state, action_batch=action_batch
        )

        # optimization objective: optimize actor to maximize Q(s, a)
        loss = -q1.mean()

        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            # sample a batch of next actions from target actor network;
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            next_action = self._actor_target.sample_action(batch.next_state)
            # (batch_size, action_dim)

            # get q values of (batch.next_state, next_action) from targets of twin critic
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            next_q1, next_q2 = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias); shape (batch_size)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.terminated.float())
            ) + batch.reward  # shape (batch_size)

        assert isinstance(self._critic, TwinCritic), "DDPG requires TwinCritic critic"

        # update twin critics towards bellman target
        loss, _, _ = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
        )

        return loss

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two DeepDeterministicPolicyGradient instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        # Inherit comparisons from the base class
        differences.append(super().compare(other))

        if not isinstance(other, DeepDeterministicPolicyGradient):
            differences.append(
                "other is not an instance of DeepDeterministicPolicyGradient"
            )

        # No additional attributes to compare in DeepDeterministicPolicyGradient

        return "\n".join(differences)
