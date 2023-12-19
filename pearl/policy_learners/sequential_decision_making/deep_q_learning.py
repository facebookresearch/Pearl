# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Optional, Tuple

import torch
from Pearl.pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from Pearl.pearl.api.action_space import ActionSpace
from Pearl.pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from Pearl.pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from Pearl.pearl.policy_learners.sequential_decision_making.deep_td_learning import (
    DeepTDLearning,
)
from Pearl.pearl.replay_buffers.transition import TransitionBatch
from Pearl.pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


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
            else EGreedyExploration(0.33),
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

        next_available_actions = next_available_actions.to(dtype=torch.int64)
        next_state = next_state.to(dtype=torch.float32)

        # display(f"{next_state.shape=} {next_available_actions.shape=}")

        # assert next_state.size(0) == self.state_dim, f"{next_state.shape=} \n {self.state_dim=}"
        # assert next_available_actions.size(0) == self.state_dim, f"{next_available_actions.shape=}"

        # if next_state.size(1) == 992:
        #    transform_layer = torch.nn.Linear(992, 122)
        #    next_state = transform_layer(next_state)

        # Flatten the next_state tensor
        # next_state_flat = next_state.view(batch_size, -1)  # Reshapes to [128, 2*994]
        # Define a transformation layer (if not already defined)
        # transform_layer = torch.nn.Linear(2*994, 124)
        # Apply the transformation
        # next_state_transformed = transform_layer(next_state_flat)  # Reshapes to [128, 124]
        # next_state_transformed = next_state_transformed.to(torch.float32)
        # Concatenate state and action batches
        # combined_batch = torch.cat([next_state, next_available_actions], dim=2)  # Shape: [124, 124]
        # Define a transformation layer
        # transform_layer = torch.nn.Linear(124, 124)  # Adjust to match the actual size
        # Apply the transformation
        # next_state_transformed = transform_layer(combined_batch)  # New shape: [124, 124]
        next_state_transformed = next_state[:, 0, :]
        next_actions_transformed = next_available_actions[:, 0, :]

        transform_layer = torch.nn.Linear(in_features=self.states_repeated_shape, out_features=self.state_dim)  # Transformation from 994 to 124 features
        next_state_transformed_input = transform_layer(next_state_transformed)  # Applying the transformation

        # display(f"{next_state_transformed.shape=}")
        next_state_action_values = self._Q_target.get_q_values(
            next_state_transformed_input, next_actions_transformed
        ).view(batch_size, -1)
        # (batch_size x action_space_size)

        next_unavailable_actions_mask = next_unavailable_actions_mask[:, 0]
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
