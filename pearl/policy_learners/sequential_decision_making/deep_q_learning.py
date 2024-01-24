# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Optional, Tuple

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.deep_td_learning import (
    DeepTDLearning,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class DeepQLearning(DeepTDLearning):
    """
    Deep Q Learning Policy Learner
    """

    def __init__(
        self,
        state_dim: int,
        learning_rate: float = 0.001,
        action_space: Optional[ActionSpace] = None,
        exploration_module: Optional[ExplorationModule] = None,
        soft_update_tau: float = 1.0,  # no soft update
        action_representation_module: Optional[ActionRepresentationModule] = None,
        **kwargs: Any,
    ) -> None:
        super(DeepQLearning, self).__init__(
            exploration_module=exploration_module
            if exploration_module is not None
            else EGreedyExploration(0.05),
            on_policy=False,
            state_dim=state_dim,
            action_space=action_space,
            learning_rate=learning_rate,
            soft_update_tau=soft_update_tau,
            action_representation_module=action_representation_module,
            **kwargs,
        )

    @torch.no_grad()
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        (
            next_state,
            next_available_actions,
            next_unavailable_actions_mask,
        ) = self._prepare_next_state_action_batch(batch)

        assert next_available_actions is not None

        next_state_action_values = self._Q_target.get_q_values(
            next_state, next_available_actions
        ).view(batch_size, -1)
        # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_unavailable_actions_mask] = -float("inf")

        # Torch.max(1) returns value, indices
        return next_state_action_values.max(1)[0]  # (batch_size)

    def _prepare_next_state_action_batch(
        self, batch: TransitionBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        assert next_state_batch is not None

        next_available_actions_batch = batch.next_available_actions
        # (batch_size x action_space_size x action_dim)

        next_unavailable_actions_mask_batch = batch.next_unavailable_actions_mask
        # (batch_size x action_space_size)

        assert isinstance(self._action_space, DiscreteActionSpace)
        number_of_actions = self._action_space.n
        next_state_batch_repeated = torch.repeat_interleave(
            next_state_batch.unsqueeze(1), number_of_actions, dim=1
        )  # (batch_size x action_space_size x state_dim)

        return (
            next_state_batch_repeated,
            next_available_actions_batch,
            next_unavailable_actions_mask_batch,
        )
