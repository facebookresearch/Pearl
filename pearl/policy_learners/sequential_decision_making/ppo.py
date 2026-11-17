# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from typing import Optional
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
from pearl.replay_buffers.transition import TransitionBatch
from torch import nn, optim
from pearl.policy_learners.sequential_decision_making.ppo_base import (
    ProximalPolicyOptimizationBase,
    PPOTransition,
    PPOTransitionBatch,
    PPOReplayBuffer,
)

__all__ = [
    "ProximalPolicyOptimization",
    "PPOTransition",
    "PPOTransitionBatch",
    "PPOReplayBuffer",
]


class ProximalPolicyOptimization(ProximalPolicyOptimizationBase):
    """Native discrete PPO with unchanged constructor and source defaults."""

    def __init__(
        self,
        action_space: ActionSpace,
        state_dim: int | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        history_summarization_learning_rate: float = 1e-4,
        exploration_module: ExplorationModule | None = None,
        actor_network_type: type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: type[ValueNetwork] = VanillaValueNetwork,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        epsilon: float = 0.0,
        trace_decay_param: float = 0.95,
        entropy_bonus_scaling: float = 0.01,
        action_representation_module: ActionRepresentationModule | None = None,
        actor_network_instance: ActorNetwork | None = None,
        critic_network_instance: ValueNetwork | nn.Module | None = None,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
        history_summarization_optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            use_critic=True,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            history_summarization_learning_rate=history_summarization_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            use_twin_critic=False,
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else PropensityExploration()
            ),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=False,
            on_policy=True,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            history_summarization_optimizer=history_summarization_optimizer,
        )
        self._epsilon = epsilon
        self._trace_decay_param = trace_decay_param
        self._entropy_bonus_scaling = entropy_bonus_scaling

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Loss = actor loss + critic loss + entropy_bonus_scaling * entropy loss
        """
        # TODO need to support continuous action
        # TODO: change the output shape of value networks
        assert isinstance(batch, PPOTransitionBatch)
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        action_probs = self._actor.get_action_prob(
            state_batch=batch.state,
            action_batch=batch.action,
            available_actions=batch.curr_available_actions,
            unavailable_actions_mask=batch.curr_unavailable_actions_mask,
        )
        # shape (batch_size)

        # actor loss
        action_probs_old = batch.action_probs
        assert action_probs_old is not None
        r_thelta = torch.div(action_probs, action_probs_old)  # shape (batch_size)
        clip = torch.clamp(
            r_thelta, min=1.0 - self._epsilon, max=1.0 + self._epsilon
        )  # shape (batch_size)
        # pyre-fixme[58]: `*` is not supported for operand types `Tensor` and
        #  `Optional[Tensor]`.
        loss = self.clipped_surrogate_loss(
            r_thelta, batch.gae, self._epsilon, reduction="sum", clipped_ratio=clip
        )
        # entropy
        entropy: torch.Tensor = torch.distributions.Categorical(
            action_probs.detach()
        ).entropy()
        loss -= torch.sum(self._entropy_bonus_scaling * entropy)
        return loss
