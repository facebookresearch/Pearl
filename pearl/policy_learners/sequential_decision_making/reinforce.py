# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Type

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.neural_networks.common.value_networks import ValueNetwork

from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym  # noqa
import torch

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
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
from pearl.replay_buffers.transition import TransitionBatch


class REINFORCE(ActorCriticBase):
    """
    Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning. Machine learning, 8, 229-256.
    The critic serves as the baseline.
    """

    def __init__(
        self,
        state_dim: int,
        actor_hidden_dims: List[int],
        critic_hidden_dims: Optional[List[int]] = None,
        action_space: Optional[ActionSpace] = None,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
        exploration_module: Optional[ExplorationModule] = None,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(REINFORCE, self).__init__(
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
            batch_size=0,  # REINFORCE does not use batch size
            is_action_continuous=False,
            on_policy=True,
            action_representation_module=action_representation_module,
        )

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = (
            batch.state
        )  # (batch_size x state_dim) note that here batch_size = episode length
        return_batch = batch.cum_reward  # (batch_size)
        policy_propensities = self._actor.get_action_prob(
            batch.state,
            batch.action,
            batch.curr_available_actions,
            batch.curr_unavailable_actions_mask,
        )  # shape (batch_size)
        negative_log_probs = -torch.log(policy_propensities + 1e-8)
        if self._use_critic:
            v = self._critic(state_batch).view(-1)  # (batch_size)
            assert return_batch is not None
            loss = torch.sum(negative_log_probs * (return_batch - v.detach()))
        else:
            loss = torch.sum(negative_log_probs * return_batch)
        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert self._use_critic, "can not compute critic loss without critic"
        assert batch.cum_reward is not None
        return single_critic_state_value_loss(
            state_batch=batch.state,
            expected_target_batch=batch.cum_reward,
            critic=self._critic,
        )
