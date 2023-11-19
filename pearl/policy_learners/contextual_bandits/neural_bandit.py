#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict, List, Optional

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.optimizers.keyed_optimizer_wrapper import (
    KeyedOptimizerWrapper,
)
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
from pearl.utils.instantiations.action_spaces.discrete import DiscreteActionSpace
from torch import optim
from torchrec.optim.keyed import CombinedOptimizer


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
        # TODO define optimizer config to use by all deep algorithms
        use_keyed_optimizer: bool = False,
        state_features_only: bool = False,
        **kwargs: Dict[str, Any]
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
        # pyre-fixme[4]: Attribute must be annotated.
        self._optimizer = optim.AdamW(
            self._deep_represent_layers.parameters(), lr=learning_rate, amsgrad=True
        )
        self._state_features_only = state_features_only
        if use_keyed_optimizer:
            optims = [
                (
                    "",
                    KeyedOptimizerWrapper(
                        models={"_deep_represent_layers.": self._deep_represent_layers},
                        optimizer_cls=optim.AdamW,
                        lr=learning_rate,
                        amsgrad=True,
                    ),
                )
            ]
            # pyre-fixme[6]: For 1st argument expected `List[Union[Tuple[str,
            #  KeyedOptimizer], KeyedOptimizer]]` but got `List[Tuple[str,
            #  KeyedOptimizerWrapper]]`.
            self._optimizer = CombinedOptimizer(optims)

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        if self._state_features_only:
            input_features = batch.state
        else:
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

    # pyre-fixme[14]: `act` overrides method defined in `ContextualBanditBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
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
        action_count = action_space.n
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,  # pyre-ignore[6]
            state_features_only=self._state_features_only,
        )
        values = self._deep_represent_layers(new_feature).squeeze()
        # batch_size * action_count
        assert values.numel() == new_feature.shape[0] * action_count
        return self._exploration_module.act(
            subjective_state=subjective_state,
            action_space=action_space,
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
        )
        batch_size = feature.shape[0]
        feature_dim = feature.shape[-1]
        # dim: [batch_size * num_arms, feature_dim]
        feature = feature.reshape(-1, feature_dim)
        # dim: [batch_size, num_arms] or [batch_size]
        return self._deep_represent_layers(feature).reshape(batch_size, -1).squeeze()

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer
