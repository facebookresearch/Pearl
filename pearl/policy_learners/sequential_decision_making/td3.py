# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, List, Optional, Type, Union

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import update_target_network
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    twin_critic_action_value_loss,
    update_critic_target_network,
)
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from torch import nn


class TD3(DeepDeterministicPolicyGradient):
    """
    TD3 uses a deterministic actor, Twin critics, and a delayed actor update.
        - An exploration module is used with deterministic actors.
        - To avoid exploration, use NoExploration module.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: Optional[List[int]] = None,
        critic_hidden_dims: Optional[List[int]] = None,
        exploration_module: Optional[ExplorationModule] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        actor_network_instance: Optional[ActorNetwork] = None,
        critic_network_instance: Optional[Union[QValueNetwork, nn.Module]] = None,
    ) -> None:
        assert isinstance(action_space, BoxActionSpace)
        super(TD3, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            exploration_module=exploration_module,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
        )
        self._action_space: BoxActionSpace = action_space
        self._actor_update_freq = actor_update_freq
        self._actor_update_noise = actor_update_noise
        self._actor_update_noise_clip = actor_update_noise_clip
        self._critic_update_count = 0
        self._last_actor_loss: float = 0.0

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # The actor and the critic updates are arranged in the following way
        # for the same reason as in the comment "If the history summarization module ..."
        # in the learn_batch function in actor_critic_base.py.

        self._critic_update_count += 1
        report = {}
        # delayed actor update
        self._history_summarization_optimizer.zero_grad()
        if self._critic_update_count % self._actor_update_freq == 0:
            self._actor_optimizer.zero_grad()
            actor_loss = self._actor_loss(batch)
            actor_loss.backward(retain_graph=True)
            self._actor_optimizer.step()
            self._last_actor_loss = actor_loss.item()
        report["actor_loss"] = self._last_actor_loss

        self._critic_optimizer.zero_grad()
        critic_loss = self._critic_loss(batch)  # critic update
        critic_loss.backward()
        self._critic_optimizer.step()
        report["critic_loss"] = critic_loss.item()
        self._history_summarization_optimizer.step()

        if self._critic_update_count % self._actor_update_freq == 0:
            # update targets of critics using soft updates
            update_critic_target_network(
                self._critic_target,
                self._critic,
                self._critic_soft_update_tau,
            )
            # update target of actor network using soft updates
            update_target_network(
                self._actor_target, self._actor, self._actor_soft_update_tau
            )

        return report

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            next_action = self._actor_target.sample_action(batch.next_state)

            # sample clipped gaussian noise
            noise = torch.normal(
                mean=0,
                std=self._actor_update_noise,
                size=next_action.size(),
                device=batch.device,
            )

            noise = torch.clamp(
                noise,
                -self._actor_update_noise_clip,
                self._actor_update_noise_clip,
            )  # shape (batch_size, action_dim)

            # rescale the noise
            low = torch.tensor(self._action_space.low, device=batch.device)
            high = torch.tensor(self._action_space.high, device=batch.device)
            noise = noise * (high - low) / 2

            # add clipped noise to next_action
            next_action = torch.clamp(
                next_action + noise, low, high
            )  # shape (batch_size, action_dim)

            # sample q values of (next_state, next_action) from targets of critics
            next_q1, next_q2 = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.terminated.float())
            ) + batch.reward  # (batch_size)

        # update twin critics towards bellman target
        assert isinstance(self._critic, TwinCritic)
        loss, _, _ = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
        )
        return loss


class TD3BC(TD3):
    """
    Implementation of the TD3BC algorithm in which a behaviour cloning term is added to the actor loss.
    The actor loss is implemented similarly to https://arxiv.org/pdf/2106.06860.pdf.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        behavior_policy: torch.nn.Module,
        actor_hidden_dims: Optional[List[int]] = None,
        critic_hidden_dims: Optional[List[int]] = None,
        exploration_module: Optional[ExplorationModule] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        actor_network_instance: Optional[ActorNetwork] = None,
        critic_network_instance: Optional[Union[QValueNetwork, nn.Module]] = None,
        alpha_bc: float = 2.5,
    ) -> None:
        super(TD3BC, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            exploration_module=exploration_module,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            actor_update_freq=actor_update_freq,
            actor_update_noise=actor_update_noise,
            actor_update_noise_clip=actor_update_noise_clip,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
        )
        self.alpha_bc: float = alpha_bc
        self._behavior_policy: torch.nn.Module = behavior_policy

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:

        # sample a batch of actions from the actor network; shape (batch_size, action_dim)
        action_batch = self._actor.sample_action(batch.state)

        # samples q values for (batch.state, action_batch) from twin critics
        q, _ = self._critic.get_q_values(
            state_batch=batch.state, action_batch=action_batch
        )

        # behvaiour cloning loss terms
        with torch.no_grad():
            behaviour_action_batch = self._behavior_policy(batch.state)
        lmbda = self.alpha_bc / q.abs().mean().detach()
        behavior_loss_mse = ((action_batch - behaviour_action_batch).pow(2)).mean()

        # optimization objective: optimize actor to maximize Q(s, a)
        loss = behavior_loss_mse - lmbda * q.mean()

        return loss
