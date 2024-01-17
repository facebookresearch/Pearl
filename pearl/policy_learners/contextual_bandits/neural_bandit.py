# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
    DEFAULT_ACTION_SPACE,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.action_utils import (
    concatenate_actions_to_state,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import optim

LOSS_TYPES = {
    "mse": torch.nn.functional.mse_loss,
    "mae": torch.nn.functional.l1_loss,
    "cross_entropy": torch.nn.functional.binary_cross_entropy,
}  # loss func for neural bandits algorithms

ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


class NeuralBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with Deep Policy.
    The MLP (_deep_represent_layers) directly returns a predicted score.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],
        exploration_module: ExplorationModule,
        output_dim: int = 1,
        training_rounds: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        state_features_only: bool = False,
        loss_type: str = "mse",  # one of the LOSS_TYPES names, e.g., mse, mae, xentropy
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(NeuralBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
        )
        self.model = VanillaValueNetwork(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        self._optimizer: torch.optim.Optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, amsgrad=True
        )
        self._state_features_only = state_features_only
        self.loss_type = loss_type

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        if self._state_features_only:
            input_features = batch.state
        else:
            input_features = torch.cat([batch.state, batch.action], dim=1)

        # forward pass
        current_values = self.model(input_features)
        expected_values = batch.reward

        criterion = LOSS_TYPES[self.loss_type]
        loss = criterion(current_values.view(expected_values.shape), expected_values)

        # Backward pass + optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit: bool = False,
    ) -> Action:
        """
        Args:
            subjective_state: state will be applied to different action vectors in action_space
            action_space: contains a list of action vector, currenly only support static space
        Return:
            action index chosen given state and action vectors
        """
        assert isinstance(available_action_space, DiscreteActionSpace)
        # It doesnt make sense to call act if we are not working with action vector
        action_count = available_action_space.n
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=available_action_space,
            state_features_only=self._state_features_only,
            action_representation_module=self._action_representation_module,
        )
        values = self.model(new_feature).squeeze()
        # batch_size * action_count
        assert values.numel() == new_feature.shape[0] * action_count
        return self._exploration_module.act(
            subjective_state=subjective_state,
            action_space=available_action_space,
            values=values,
            action_availability_mask=action_availability_mask,
            representation=None,  # fill in as needed in the future
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = DEFAULT_ACTION_SPACE,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state: tensor for state
            action_space: basically a list of action features, when it is none, view
                subjective_state as feature
        Return:
            return mlp value with shape (batch_size, action_count)
        """
        feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            state_features_only=self._state_features_only,
            action_representation_module=self._action_representation_module,
        )
        batch_size = feature.shape[0]
        feature_dim = feature.shape[-1]
        # dim: [batch_size * num_arms, feature_dim]
        feature = feature.reshape(-1, feature_dim)
        # dim: [batch_size, num_arms] (or [batch_size] if num_arms==1)
        return self.model(feature).reshape(batch_size, -1).squeeze()

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer
