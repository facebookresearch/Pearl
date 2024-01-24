# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Type

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaActorNetwork,
)

from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
    twin_critic_action_value_loss,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


# Currently available actions is not used. Needs to be updated once we know the input
# structure of production stack on this param.

# TODO: to make things easier with a single optimizer, we need to polish this method.
class SoftActorCritic(ActorCriticBase):
    """
    Implementation of Soft Actor Critic Policy Learner for discrete action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        critic_soft_update_tau: float = 0.005,
        exploration_module: Optional[ExplorationModule] = None,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        entropy_coef: float = 0.2,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(SoftActorCritic, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=True,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=critic_soft_update_tau,
            use_twin_critic=True,
            exploration_module=exploration_module
            if exploration_module is not None
            else PropensityExploration(),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=False,
            on_policy=False,
            action_representation_module=action_representation_module,
        )

        # This is needed to avoid actor softmax overflow issue.
        # Should not be left for users to choose.
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self._actor_optimizer, gamma=0.99
        )

        # TODO: implement learnable entropy coefficient
        self._entropy_coef = entropy_coef

    # sac uses a learning rate scheduler specifically
    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space
        self.scheduler.step()

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:

        reward_batch = batch.reward  # (batch_size)
        done_batch = batch.done  # (batch_size)

        assert done_batch is not None
        expected_state_action_values = (
            self._get_next_state_expected_values(batch)
            * self._discount_factor
            * (1 - done_batch.float())
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        assert isinstance(self._critic, TwinCritic)
        loss = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
        )

        return loss

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_available_actions_batch = (
            batch.next_available_actions
        )  # (batch_size x action_space_size x action_dim)
        next_unavailable_actions_mask_batch = (
            batch.next_unavailable_actions_mask
        )  # (batch_size x action_space_size)

        assert next_state_batch is not None
        assert next_available_actions_batch is not None
        next_state_batch_repeated = torch.repeat_interleave(
            next_state_batch.unsqueeze(1),
            self.action_representation_module.max_number_actions,
            dim=1,
        )  # (batch_size x action_space_size x state_dim)

        # get q values of (states, all actions) from twin critics
        next_q1, next_q2 = self._critic_target.get_q_values(
            state_batch=next_state_batch_repeated,
            action_batch=next_available_actions_batch,
        )

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.minimum(next_q1, next_q2)

        # random ensemble distillation (reduce overestimation bias)
        # random_index = torch.randint(0, 2, (1,)).item()
        # next_q = next_q1 if random_index == 0 else next_q2

        next_state_action_values = next_q.view(
            self.batch_size, -1
        )  # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to 0.0
        # since we are calculating expectation

        if next_unavailable_actions_mask_batch is not None:
            next_state_action_values[next_unavailable_actions_mask_batch] = 0.0

        next_state_policy_dist = self._actor.get_policy_distribution(
            state_batch=next_state_batch,
            available_actions=next_available_actions_batch,
            unavailable_actions_mask=next_unavailable_actions_mask_batch,
        )  # (batch_size x action_space_size)

        # Entropy Regularization
        next_state_action_values = (
            next_state_action_values
            - self._entropy_coef * torch.log(next_state_policy_dist + 1e-8)
        ) * next_state_policy_dist  # (batch_size x action_space_size)

        return next_state_action_values.sum(dim=1)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state  # (batch_size x state_dim)
        state_batch_repeated = torch.repeat_interleave(
            state_batch.unsqueeze(1),
            self.action_representation_module.max_number_actions,
            dim=1,
        )  # (batch_size x action_space_size x state_dim)

        available_actions = (
            batch.curr_available_actions
        )  # (batch_size x action_space_size x action_dim)

        # get q values of (states, all actions) from twin critics
        q1, q2 = self._critic.get_q_values(
            state_batch=state_batch_repeated, action_batch=available_actions
        )
        # clipped double q learning (reduce overestimation bias)
        q = torch.minimum(q1, q2)

        unavailable_actions_mask = (
            batch.curr_unavailable_actions_mask
        )  # (batch_size x action_space_size)

        new_policy_dist = self._actor.get_policy_distribution(
            state_batch=state_batch,
            available_actions=available_actions,
            unavailable_actions_mask=unavailable_actions_mask,
        )  # (batch_size x action_space_size)

        state_action_values = q.view(
            (self.batch_size, self.action_representation_module.max_number_actions)
        )  # (batch_size x action_space_size)

        if unavailable_actions_mask is not None:
            state_action_values[unavailable_actions_mask] = 0.0

        loss = (
            (
                new_policy_dist
                * (
                    self._entropy_coef * torch.log(new_policy_dist + 1e-8)
                    - state_action_values
                )
            )
            .sum(dim=1)
            .mean()
        )

        return loss
