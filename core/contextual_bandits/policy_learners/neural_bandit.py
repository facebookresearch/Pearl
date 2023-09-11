#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict, List

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.contextual_bandits.policy_learners.contextual_bandit_base import (
    ContextualBanditBase,
)
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace
from torch import optim
from torch.nn.parallel import DistributedDataParallel


class NeuralBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with Deep Policy
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
        **kwargs,
    ) -> None:
        super(NeuralBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        self._deep_represent_layers = VanillaValueNetwork(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs,
        )
        if self.distribution_enabled:
            self._deep_represent_layers.to(self.device)
            self._deep_represent_layers = DistributedDataParallel(
                self._deep_represent_layers,
                device_ids=None if self.device.type == "cpu" else [self.device],
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
        """
        Args:
            subjective_state - state will be applied to different action vectors in action_space
            action_space contains a list of action vector, currenly only support static space
        Return:
            action index chosen given state and action vectors
        """
        # It doesnt make sense to call act if we are not working with action vector
        assert action_space.action_dim > 0
        action_count = action_space.n
        new_feature = action_space.cat_state_tensor(subjective_state)
        values = self._deep_represent_layers(new_feature).squeeze()
        # batch_size * action_count
        assert values.numel() == new_feature.shape[0] * action_count
        return self._exploration_module.act(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=None,  # fill in as needed in the future
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state: tensor for state
            action_space: basically a list of action features, when it is none, view subjective_state as feature
        Return:
            return mlp value with shape (batch_size, action_count)
        """
        feature = (
            action_space.cat_state_tensor(subjective_state)
            if action_space is not None
            else subjective_state
        )
        return self._deep_represent_layers(feature).squeeze()

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "deep_represent_layers": self._deep_represent_layers.state_dict(),
        }

    def set_model_state(self, state: Dict[str, Any], strict=True):
        self._deep_represent_layers.load_state_dict(state["deep_represent_layers"])

    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        return [self._optimizer]
