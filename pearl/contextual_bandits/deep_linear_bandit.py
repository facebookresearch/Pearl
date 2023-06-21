#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict, List

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.contextual_bandits.deep_bandit import DeepBandit
from pearl.contextual_bandits.linear_regression import AvgWeightLinearRegression
from pearl.contextual_bandits.linucb_exploration import LinUCBExploration
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch
from pearl.utils.action_spaces import DiscreteActionSpace


class DeepLinearBandit(DeepBandit):
    """
    Policy Learner for Contextual Bandit with:
    features --> neural networks --> linear regression --> predicted rewards
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],  # last one is the input dim for linear regression
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
    ) -> None:
        assert (
            len(hidden_dims) >= 1
        ), "hidden_dims should have at least one value to specify feature dim for linear regression"
        DeepBandit.__init__(
            self,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        # TODO specify linear regression type when needed
        self._linear_regression = AvgWeightLinearRegression(feature_dim=hidden_dims[-1])
        self._linear_regression_dim = hidden_dims[-1]

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        input_features = torch.cat([batch.state, batch.action], dim=1)

        # forward pass
        mlp_output = self._deep_represent_layers(input_features)
        current_values = self._linear_regression(mlp_output)
        expected_values = batch.reward

        criterion = torch.nn.MSELoss()
        loss = criterion(current_values.view(expected_values.shape), expected_values)

        # Optimize the deep layer
        # TODO how should we handle weight in NN training
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # Optimize linear regression
        self._linear_regression.train(
            mlp_output.detach(), expected_values, batch.weight
        )
        return {"mlp_loss": loss.item()}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # It doesnt make sense to call act if we are not working with action vector
        assert action_space.action_dim > 0

        new_feature = action_space.cat_state_tensor(subjective_state)
        mlp_values = self._deep_represent_layers(new_feature)
        # `_linear_regression` is not nn.Linear(). It is a customized linear layer
        # that can be updated by analytical method (matrix calculations) rather than gradient descent of torch optimizer.
        values = self._linear_regression(mlp_values)

        # batch_size * action_count
        assert values.numel() == new_feature.shape[0] * action_space.n

        # subjective_state=mlp_values makes sense for LinUCBExploration
        # revisit for other exploration module
        return self._exploration_module.act(
            subjective_state=mlp_values,
            action_space=action_space,
            values=values,
            representation=self._linear_regression,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = None,
    ) -> torch.Tensor:
        # TODO generalize for all kinds of exploration module
        assert isinstance(self._exploration_module, LinUCBExploration)
        feature = (
            action_space.cat_state_tensor(subjective_state)
            if action_space is not None
            else subjective_state
        )
        processed_feature = self._deep_represent_layers(feature)
        return self._exploration_module.get_ucb_scores(
            subjective_state=processed_feature,
            values=self._linear_regression(processed_feature),
            # when action_space is None, we are querying score for one action
            available_action_space=action_space
            if action_space is not None
            else DiscreteActionSpace([0]),
            representation=self._linear_regression,
        ).squeeze()
