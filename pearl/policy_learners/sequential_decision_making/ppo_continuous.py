# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import torch
from typing import Any, Dict, List, Optional, Type, Union

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,GaussianActorNetwork
)
from pearl.policy_learners.exploration_modules.common import NoExploration

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)

from pearl.policy_learners.sequential_decision_making.ppo_base import (
    ProximalPolicyOptimizationBase,
    PPOTransitionBatch,
)

from pearl.replay_buffers.transition import TransitionBatch
from pearl.policy_learners.sequential_decision_making.ppo_base import ProximalPolicyOptimizationBase

from torch import nn

class ContinuousProximalPolicyOptimization(ProximalPolicyOptimizationBase):
    """
    PPO algorithm for continuous action spaces.
    """

    def __init__(
            self,
            state_dim: int,
            action_space: ActionSpace,
            use_critic: bool,
            actor_hidden_dims: Optional[List[int]] = None,
            critic_hidden_dims: Optional[List[int]] = None,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-4,
            exploration_module: Optional[ExplorationModule] = None,
            actor_network_type: Type[ActorNetwork] = GaussianActorNetwork,
            critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
            discount_factor: float = 0.99,
            training_rounds: int = 100,
            batch_size: int = 128,
            epsilon: float = 0.2,
            trace_decay_param: float = 0.95,
            entropy_bonus_scaling: float = 0.01,
            normalize_gae: bool = True,
            action_representation_module: Optional[ActionRepresentationModule] = None,
            actor_network_instance: Optional[ActorNetwork] = None,
            critic_network_instance: Optional[Union[ValueNetwork, nn.Module]] = None,
    ) -> None:
        super(ContinuousProximalPolicyOptimization, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            use_critic=use_critic,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else NoExploration()
            ),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=True,  # Change to continuous action space
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
        )

        self._normalize_gae = normalize_gae
        self._epsilon = epsilon
        self._trace_decay_param = trace_decay_param
        self._entropy_bonus_scaling = entropy_bonus_scaling

        assert self.is_action_continuous is True

    def _get_action_prob(self, history_summary_batch, action_representation_batch) -> torch.Tensor:
          action_probs = (
                self._actor.get_log_probability(
                    state_batch=history_summary_batch,
                    action_batch=action_representation_batch,
                ).detach().unsqueeze(-1))
          return action_probs

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Loss = actor loss + critic loss + entropy_bonus_scaling * entropy loss
        """
        assert isinstance(batch, PPOTransitionBatch)

        log_action_probs, normal = self._actor.get_log_probability(
            state_batch=batch.state,
            action_batch=batch.action, get_distribution=True
        )

        log_action_probs = torch.squeeze(log_action_probs)

        # Calculate the log ratio for PPO
        log_action_probs_old = torch.squeeze(batch.action_probs)
        assert log_action_probs_old is not None

        # Use log difference for stability
        r_theta = torch.exp(log_action_probs - log_action_probs_old)  # shape (batch_size)
        clip = torch.clamp(r_theta, min=1.0 - self._epsilon, max=1.0 + self._epsilon)

        gae = batch.gae
        if self._normalize_gae:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        # clipped surrogate loss
        clip_loss_1 = gae * r_theta
        clip_loss_2 = gae * clip
        loss = -torch.min(clip_loss_1, clip_loss_2)
        loss = loss.mean()

        # Entropy for encouraging exploration
        entropy = normal.entropy().sum(axis=-1).mean()
        loss -= self._entropy_bonus_scaling * entropy

        return loss