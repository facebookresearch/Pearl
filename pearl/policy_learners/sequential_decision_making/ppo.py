# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

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
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch

from pearl.utils.functional_utils.learning.critic_utils import (
    single_critic_state_value_loss,
)
from pearl.utils.replay_buffer_utils import (
    make_replay_buffer_class_for_specific_transition_types,
)
from torch import nn


@dataclass(frozen=False)
class PPOTransition(Transition):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_probs: Optional[torch.Tensor] = None  # action probs


@dataclass(frozen=False)
class PPOTransitionBatch(TransitionBatch):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_probs: Optional[torch.Tensor] = None  # action probs

    @classmethod
    def from_parent(
        cls,
        parent_obj: TransitionBatch,
        gae: Optional[torch.Tensor] = None,
        lam_return: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None,
    ) -> "PPOTransitionBatch":
        # Extract attributes from parent_obj using __dict__ and create a new child object
        child_obj = cls(
            **parent_obj.__dict__,
            gae=gae,
            lam_return=lam_return,
            action_probs=action_probs,
        )
        return child_obj


PPOReplayBuffer: Type[TensorBasedReplayBuffer] = (
    make_replay_buffer_class_for_specific_transition_types(
        PPOTransition, PPOTransitionBatch
    )
)


class ProximalPolicyOptimization(ActorCriticBase):
    """
    paper: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: Optional[List[int]] = None,
        critic_hidden_dims: Optional[List[int]] = None,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        history_summarization_learning_rate: float = 1e-4,
        exploration_module: Optional[ExplorationModule] = None,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        epsilon: float = 0.0,
        trace_decay_param: float = 0.95,
        entropy_bonus_scaling: float = 0.01,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        actor_network_instance: Optional[ActorNetwork] = None,
        critic_network_instance: Optional[Union[ValueNetwork, nn.Module]] = None,
    ) -> None:
        super(ProximalPolicyOptimization, self).__init__(
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
        loss = torch.sum(-torch.min(r_thelta * batch.gae, clip * batch.gae))
        # entropy
        entropy: torch.Tensor = torch.distributions.Categorical(
            action_probs.detach()
        ).entropy()
        loss -= torch.sum(self._entropy_bonus_scaling * entropy)
        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert isinstance(batch, PPOTransitionBatch)
        assert batch.lam_return is not None
        return single_critic_state_value_loss(
            state_batch=batch.state,
            expected_target_batch=batch.lam_return,
            critic=self._critic,
        )

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        self.preprocess_replay_buffer(replay_buffer)
        # sample from replay buffer and learn
        result = super().learn(replay_buffer)
        # update old actor with latest actor for next round
        return result

    def preprocess_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
        """
        Preprocess the replay buffer by calculating
        and adding the generalized advantage estimates (gae),
        truncated lambda returns (lam_return) and action probabilities (action_probs)
        under the current policy.
        See https://arxiv.org/abs/1707.06347 equation (11) for the definition of gae.
        See "Reinforcement Learning: An Introduction" by Sutton and Barto (2018) equation (12.10)
        for the definition of truncated lambda return.
        """
        assert isinstance(replay_buffer, TensorBasedReplayBuffer)
        assert len(replay_buffer.memory) > 0
        (
            state_list,
            action_list,
            available_actions_list,
            unavailable_actions_mask_list,
        ) = ([], [], [], [])
        for transition in reversed(replay_buffer.memory):
            state_list.append(transition.state)
            action_list.append(transition.action)
            available_actions_list.append(transition.curr_available_actions)
            unavailable_actions_mask_list.append(
                transition.curr_unavailable_actions_mask
            )
        history_summary_batch = self._history_summarization_module(
            torch.cat(state_list)
        ).detach()
        action_representation_batch = self._action_representation_module(
            torch.cat(action_list)
        )

        # Transitions in the reply buffer memory are in the CPU
        # (only sampled batches are moved to the used device, kept in replay_buffer.device)
        # To use it in expressions involving the models,
        # we must move them to the device being used first.
        history_summary_batch = history_summary_batch.to(
            replay_buffer.device_for_batches
        )
        action_representation_batch = action_representation_batch.to(
            replay_buffer.device_for_batches
        )

        state_values = self._critic(history_summary_batch).detach()
        action_probs = (
            self._actor.get_action_prob(
                state_batch=history_summary_batch,
                action_batch=action_representation_batch,
            )
            .detach()
            .unsqueeze(-1)
        )

        # Transitions in the reply buffer memory are in the CPU
        # (only sampled batches are moved to the used device,
        # kept in replay_buffer.device_for_batches)
        # To use it in expressions involving the critic,
        # we must move them to the device being used first.
        next_state = replay_buffer.memory[-1].next_state
        assert next_state is not None
        next_state_in_device = next_state.to(replay_buffer.device_for_batches)

        # Obtain the value of the most recent state stored in the replay buffer.
        # This value is used to compute the generalized advantage estimation (gae)
        # and the truncated lambda return for all states in the replay buffer.
        next_value = self._critic(
            self._history_summarization_module(next_state_in_device)
        ).detach()[
            0
        ]  # shape (1,)
        gae = torch.tensor([0.0]).to(state_values.device)
        for i, transition in enumerate(reversed(replay_buffer.memory)):
            original_transition_device = transition.device
            transition.to(state_values.device)
            td_error = (
                transition.reward
                + self._discount_factor * next_value * (~transition.terminated)
                - state_values[i]
            )
            gae = (
                td_error
                + self._discount_factor
                * self._trace_decay_param
                * (~transition.terminated)
                * gae
            )
            assert isinstance(transition, PPOTransition)
            transition.gae = gae
            # truncated lambda return of the state
            transition.lam_return = gae + state_values[i]
            # action probabilities from the current policy
            transition.action_probs = action_probs[i]
            next_value = state_values[i]
            transition.to(original_transition_device)
