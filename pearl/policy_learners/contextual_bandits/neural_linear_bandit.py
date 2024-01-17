# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.contextual_bandit.neural_linear_regression import (
    NeuralLinearRegression,
)
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
    DEFAULT_ACTION_SPACE,
)
from pearl.policy_learners.contextual_bandits.neural_bandit import LOSS_TYPES
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
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


class NeuralLinearBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with:
    features --> neural networks --> linear regression --> predicted rewards

    The difference vs its parent class NeuralBandit is the extra
    linear regression on top of `_deep_represent_layers`.
    Here _deep_represent_layers can be treated as featuer processing,
    and then processed features are fed into a linear regression layer to output predicted score.
    For example : features --> neural networks --> LinUCB --> UCB score
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],  # last one is the input dim for linear regression
        exploration_module: Optional[ExplorationModule] = None,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        training_rounds: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.0003,
        l2_reg_lambda_linear: float = 1.0,
        state_features_only: bool = False,
        loss_type: str = "mse",  # one of the LOSS_TYPES names, e.g., mse, mae, xentropy
        output_activation_name: str = "linear",
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        hidden_activation: str = "relu",
        last_activation: Optional[str] = None,
        dropout_ratio: float = 0.0,
        use_skip_connections: bool = False,
    ) -> None:
        assert (
            len(hidden_dims) >= 1
        ), "hidden_dims should have at least one value to specify feature dim for linear regression"
        super(NeuralLinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
        )
        self.model = NeuralLinearRegression(
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            l2_reg_lambda_linear=l2_reg_lambda_linear,
            output_activation_name=output_activation_name,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            hidden_activation=hidden_activation,
            last_activation=last_activation,
            dropout_ratio=dropout_ratio,
            use_skip_connections=use_skip_connections,
        )
        self._optimizer: torch.optim.Optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, amsgrad=True
        )
        self._state_features_only = state_features_only
        self.loss_type = loss_type

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        if self._state_features_only:
            input_features = batch.state
        else:
            input_features = torch.cat([batch.state, batch.action], dim=1)

        # forward pass
        model_ret = self.model.forward_with_intermediate_values(input_features)
        predicted_values = model_ret["pred_label"]
        expected_values = batch.reward
        batch_weight = batch.weight

        # criterion = mae, mse, Xentropy
        # Xentropy loss apply Sigmoid, MSE or MAE apply Identiy
        criterion = LOSS_TYPES[self.loss_type]
        if self.loss_type == "cross_entropy":
            assert torch.all(expected_values >= 0) and torch.all(expected_values <= 1)
            assert torch.all(predicted_values >= 0) and torch.all(predicted_values <= 1)
            assert isinstance(self.model.output_activation, torch.nn.Sigmoid)

        # TODO: handle weight in NN training by computing weighted loss
        loss = criterion(predicted_values.view(expected_values.shape), expected_values)

        # Optimize the NN via backpropagation
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # Optimize linear regression
        self.model._linear_regression_layer.learn_batch(
            model_ret["nn_output"].detach(),
            expected_values,
            batch_weight,
        )
        return {
            "mlp_loss": loss.item(),
            "current_values": predicted_values.mean().item(),
        }

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit: bool = False,
    ) -> Action:
        assert isinstance(available_action_space, DiscreteActionSpace)
        # It doesnt make sense to call act if we are not working with action vector
        assert isinstance(available_action_space, DiscreteActionSpace)
        assert available_action_space.action_dim > 0
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=available_action_space,
            state_features_only=self._state_features_only,
            action_representation_module=self._action_representation_module,
        )
        model_ret = self.model.forward_with_intermediate_values(new_feature)
        values = model_ret["pred_label"]

        # batch_size * action_count
        assert values.numel() == new_feature.shape[0] * available_action_space.n

        # subjective_state=mlp_values because uncertainty is only measure in the output linear layer
        # revisit for other exploration module
        return self._exploration_module.act(
            subjective_state=model_ret["nn_output"],
            action_space=available_action_space,
            values=values,
            action_availability_mask=action_availability_mask,
            representation=self.model._linear_regression_layer,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = DEFAULT_ACTION_SPACE,
    ) -> torch.Tensor:
        # TODO generalize for all kinds of exploration module
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
        # dim: [batch_size, num_arms, feature_dim]
        model_ret = self.model.forward_with_intermediate_values(feature)
        # dim: [batch_size * num_arms, 1]
        assert isinstance(self._exploration_module, UCBExploration)
        scores = self._exploration_module.get_scores(
            subjective_state=model_ret["nn_output"],
            values=model_ret["pred_label"],
            action_space=action_space,
            representation=self.model._linear_regression_layer,
        )
        # dim: [batch_size, num_arms] or [batch_size]
        scores = self.model.output_activation(scores)
        return scores.reshape(batch_size, -1).squeeze()
