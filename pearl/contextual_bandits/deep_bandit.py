#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict, List

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.contextual_bandits.contextual_bandit_base import ContextualBanditBase
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.value_networks import VanillaValueNetwork
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch
from torch import optim


class DeepBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with Deep Policy
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
    ) -> None:
        super(DeepBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        self._deep_represent_layers = VanillaValueNetwork(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
        )
        self._optimizer = optim.AdamW(
            self._deep_represent_layers.parameters(), lr=learning_rate, amsgrad=True
        )

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        input_features = torch.cat([batch.state, batch.action], dim=1)

        # forward pass
        current_values = self._deep_represent_layers(input_features)
        expected_values = batch.reward

        criterion = torch.nn.MSELoss()
        loss = criterion(current_values.view(expected_values.shape), expected_values)

        # Optimize the deep layer
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return {"loss": loss.item()}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        raise NotImplementedError("Implement when there is a usecase")

    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> torch.Tensor:
        return self._deep_represent_layers(subjective_state).squeeze()
