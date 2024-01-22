# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from typing import Any, Dict, List, Optional, Type

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaActorNetwork,
)
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
    single_critic_state_value_loss,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from torch import nn


class ProximalPolicyOptimization(ActorCriticBase):
    """
    paper: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: List[int],
        critic_hidden_dims: Optional[List[int]],
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        exploration_module: Optional[ExplorationModule] = None,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        epsilon: float = 0.0,
        entropy_bonus_scaling: float = 0.01,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(ProximalPolicyOptimization, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            use_twin_critic=False,
            exploration_module=exploration_module
            if exploration_module is not None
            else PropensityExploration(),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=False,
            on_policy=True,
            action_representation_module=action_representation_module,
        )
        self._epsilon = epsilon
        self._entropy_bonus_scaling = entropy_bonus_scaling
        self._actor_old: nn.Module = copy.deepcopy(self._actor)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Loss = actor loss + critic loss + entropy_bonus_scaling * entropy loss
        """
        # TODO: change the output shape of value networks
        vs: torch.Tensor = self._critic(batch.state).view(-1)  # shape (batch_size)
        action_probs = self._actor.get_action_prob(
            state_batch=batch.state,
            action_batch=batch.action,
            available_actions=batch.curr_available_actions,
            unavailable_actions_mask=batch.curr_unavailable_actions_mask,
        )
        # shape (batch_size)

        # actor loss
        with torch.no_grad():
            action_probs_old = self._actor_old.get_action_prob(
                state_batch=batch.state,
                action_batch=batch.action,
                available_actions=batch.curr_available_actions,
                unavailable_actions_mask=batch.curr_unavailable_actions_mask,
            )  # shape (batch_size)
        r_thelta = torch.div(action_probs, action_probs_old)  # shape (batch_size)
        clip = torch.clamp(
            r_thelta, min=1.0 - self._epsilon, max=1.0 + self._epsilon
        )  # shape (batch_size)

        # advantage estimator, in paper:
        # A = sum(lambda^t*gamma^t*TD_error), while TD_error = reward + gamma * V(s+1) - V(s)
        # when lambda = 1 and gamma = 1
        # A = sum(TD_error) = return - V(s)
        # TODO support lambda and gamma
        with torch.no_grad():
            advantage = batch.cum_reward - vs  # shape (batch_size)

        # entropy
        # Categorical is good for Cartpole Env where actions are discrete
        # TODO need to support continuous action
        entropy: torch.Tensor = torch.distributions.Categorical(
            action_probs.detach()
        ).entropy()
        loss = torch.sum(
            -torch.min(r_thelta * advantage, clip * advantage)
        ) - torch.sum(self._entropy_bonus_scaling * entropy)
        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert batch.cum_reward is not None
        return single_critic_state_value_loss(
            state_batch=batch.state,
            expected_target_batch=batch.cum_reward,
            critic=self._critic,
        )

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        result = super().learn(replay_buffer)
        # update old actor with latest actor for next round
        self._actor_old.load_state_dict(self._actor.state_dict())
        return result
