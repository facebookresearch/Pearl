# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, List, Optional, Union

import torch

import torch.nn as nn

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    DEFAULT_ACTION_SPACE,
)
from pearl.policy_learners.contextual_bandits.neural_bandit import (
    ACTIVATION_MAP,
    LOSS_TYPES,
    NeuralBandit,
)
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
from pearl.utils.functional_utils.learning.linear_regression import LinearRegression
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class NeuralLinearBandit(NeuralBandit):
    """
    Policy Learner for Contextual Bandit with:
    features --> neural networks --> linear regression --> predicted rewards

    The difference vs its parent class NeuralBandit is the extra
    linear regression on top of `_deep_represent_layers`.
    Here _deep_represent_layers can be treated as featuer processing,
    and then processed features are fed into a linear regression layer to output predicted score.
    For example : features --> neural networks --> LinUCB --> UCB score
    """

    output_activation: Union[
        nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Softplus, nn.Tanh, nn.Identity
    ]

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],  # last one is the input dim for linear regression
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        l2_reg_lambda_linear: float = 1.0,
        state_features_only: bool = False,
        loss_type: str = "mse",  # one of the LOSS_TYPES names, e.g., mse, mae, xentropy
        output_activation_name: str = "linear",
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        assert (
            len(hidden_dims) >= 1
        ), "hidden_dims should have at least one value to specify feature dim for linear regression"
        super(NeuralLinearBandit, self).__init__(
            feature_dim=feature_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            state_features_only=state_features_only,
            loss_type=loss_type,
            **kwargs,
        )
        # TODO specify linear regression type when needed
        self._linear_regression = LinearRegression(
            feature_dim=hidden_dims[-1],
            l2_reg_lambda=l2_reg_lambda_linear,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self._linear_regression_dim = hidden_dims[-1]
        self.loss_type = loss_type
        self.output_activation = ACTIVATION_MAP[output_activation_name]()
        self.output_activation_name = output_activation_name

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        if self._state_features_only:
            input_features = batch.state
        else:
            input_features = torch.cat([batch.state, batch.action], dim=1)

        # forward pass
        mlp_output = self._deep_represent_layers(input_features)
        lr_output = self._linear_regression(mlp_output)
        expected_values = batch.reward

        # criterion = mae, mse, Xentropy
        # Xentropy loss apply Sigmoid, MSE or MAE apply Identiy
        criterion = LOSS_TYPES[self.loss_type]
        current_values = self.output_activation(lr_output)
        if self.loss_type == "cross_entropy":
            assert torch.all(expected_values >= 0) and torch.all(expected_values <= 1)
            assert torch.all(current_values >= 0) and torch.all(current_values <= 1)
            assert self.output_activation_name == "sigmoid"
        loss = criterion(current_values.view(expected_values.shape), expected_values)

        # Optimize the deep layer
        # TODO how should we handle weight in NN training
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # Optimize linear regression
        batch_weight = batch.weight
        self._linear_regression.learn_batch(
            mlp_output.detach(),
            expected_values,
            batch_weight,
        )
        return {"mlp_loss": loss.item(), "current_values": current_values.mean().item()}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit: bool = False,
    ) -> Action:
        assert isinstance(action_space, DiscreteActionSpace)
        # It doesnt make sense to call act if we are not working with action vector
        assert isinstance(action_space, DiscreteActionSpace)
        assert action_space.action_dim > 0
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            state_features_only=self._state_features_only,
        )
        mlp_values = self._deep_represent_layers(new_feature)
        # `_linear_regression` is not nn.Linear().
        # It is a customized linear layer
        # that can be updated by analytical method (matrix calculations)
        # rather than gradient descent of torch optimizer.
        values = self._linear_regression(mlp_values)

        # batch_size * action_count
        assert values.numel() == new_feature.shape[0] * action_space.n

        # subjective_state=mlp_values because uncertainty is only measure in the output linear layer
        # revisit for other exploration module
        return self._exploration_module.act(
            subjective_state=mlp_values,
            action_space=action_space,
            values=values,
            action_availability_mask=action_availability_mask,
            representation=self._linear_regression,
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
        )
        batch_size = feature.shape[0]
        feature_dim = feature.shape[-1]
        # dim: [batch_size * num_arms, feature_dim]
        feature = feature.reshape(-1, feature_dim)
        # dim: [batch_size, num_arms, feature_dim]
        processed_feature = self._deep_represent_layers(feature)
        # dim: [batch_size * num_arms, 1]
        assert isinstance(self._exploration_module, UCBExploration)
        scores = self._exploration_module.get_scores(
            subjective_state=processed_feature,
            values=self._linear_regression(processed_feature),
            action_space=action_space,
            representation=self._linear_regression,
        )
        # dim: [batch_size, num_arms] or [batch_size]
        scores = self.output_activation(scores)
        return scores.reshape(batch_size, -1).squeeze()
