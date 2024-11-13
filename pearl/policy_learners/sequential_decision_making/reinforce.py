# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.neural_networks.common.value_networks import ValueNetwork

from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition
from pearl.utils.replay_buffer_utils import (
    make_replay_buffer_class_for_specific_transition_types,
)
from torch import nn

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
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    single_critic_state_value_loss,
)


@dataclass(frozen=False)
class REINFORCETransition(Transition):
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward


@dataclass(frozen=False)
class REINFORCETransitionBatch(TransitionBatch):
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward

    @classmethod
    def from_parent(
        cls,
        parent_obj: TransitionBatch,
        cum_reward: Optional[torch.Tensor] = None,
    ) -> "REINFORCETransitionBatch":
        # Extract attributes from parent_obj using __dict__ and create a new child object
        child_obj = cls(
            **parent_obj.__dict__,
            cum_reward=cum_reward,
        )
        return child_obj


REINFORCEReplayBuffer: Type[TensorBasedReplayBuffer] = (
    make_replay_buffer_class_for_specific_transition_types(
        REINFORCETransition, REINFORCETransitionBatch
    )
)


class REINFORCE(ActorCriticBase):
    """
    Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning. Machine learning, 8, 229-256.
    The critic serves as the baseline.
    """

    def __init__(
        self,
        state_dim: int,
        actor_hidden_dims: Optional[List[int]] = None,
        use_critic: bool = False,
        critic_hidden_dims: Optional[List[int]] = None,
        action_space: Optional[ActionSpace] = None,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        history_summarization_learning_rate: float = 1e-4,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
        exploration_module: Optional[ExplorationModule] = None,
        discount_factor: float = 0.99,
        training_rounds: int = 8,
        batch_size: int = 64,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        actor_network_instance: Optional[ActorNetwork] = None,
        critic_network_instance: Optional[Union[ValueNetwork, nn.Module]] = None,
    ) -> None:
        super(REINFORCE, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            use_critic=use_critic,
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

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert isinstance(batch, REINFORCETransitionBatch)
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
        assert isinstance(batch, REINFORCETransitionBatch)
        assert batch.cum_reward is not None
        return single_critic_state_value_loss(
            state_batch=batch.state,
            expected_target_batch=batch.cum_reward,
            critic=self._critic,
        )

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        assert type(replay_buffer) is REINFORCEReplayBuffer
        assert len(replay_buffer.memory) > 0
        # compute return for all states in the buffer

        # Transitions in the reply buffer memory are in the CPU
        # (only sampled batches are moved to the used device,
        # kept in replay_buffer.device_for_batches)
        # To use it in expressions involving the critic,
        # we must move them to the device being used first.
        next_state = replay_buffer.memory[-1].next_state
        terminated = replay_buffer.memory[-1].terminated
        assert next_state is not None
        assert terminated is not None
        next_state_in_device = next_state.to(replay_buffer.device_for_batches)
        terminated_in_device = terminated.to(replay_buffer.device_for_batches)

        cum_reward = self._critic(
            self._history_summarization_module(next_state_in_device)
        ).detach() * (~terminated_in_device)

        # move cum_reward to CPU to process CPU-stored transitions
        cum_reward = cum_reward.cpu()
        for transition in reversed(replay_buffer.memory):
            cum_reward += transition.reward
            assert isinstance(transition, REINFORCETransition)
            transition.cum_reward = cum_reward
        # sample from replay buffer and learn
        result = super().learn(replay_buffer)
        return result
