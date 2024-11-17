# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, List, Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
    SubjectiveState,
)
from pearl.neural_networks.common.utils import ensemble_forward

from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
)
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.action_utils import (
    concatenate_actions_to_state,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class DisjointBanditContainer(ContextualBanditBase):
    """
    Wrapper for disjoint models with discrete (and usually small) action space.
    Each action has its own bandit model (can be based on UCB or Thompson Sampling).
    Using the Composite design pattern:
    https://refactoring.guru/design-patterns/composite
    """

    def __init__(
        self,
        feature_dim: int,
        arm_bandits: List[ContextualBanditBase],
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
        state_features_only: bool = False,
    ) -> None:
        super(DisjointBanditContainer, self).__init__(
            feature_dim=feature_dim,
            exploration_module=exploration_module,
            training_rounds=training_rounds,
            batch_size=batch_size,
        )
        # Currently our disjoint LinUCB usecase only use LinearRegression
        self._arm_bandits: torch.nn.ModuleList = torch.nn.ModuleList(arm_bandits)
        self._n_arms: int = len(arm_bandits)
        self._state_features_only = state_features_only
        self._null_batch: Optional[TransitionBatch] = None

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def _validate_batch(self, batch: TransitionBatch) -> None:
        assert (
            batch.action.dtype == torch.long
        ), "action must be torch.long type (index of arm)"
        assert batch.action.min().item() >= 0, "action must be >= 0"
        assert (
            batch.action.max().item() < self._n_arms
        ), "action must be < number of arms"

    def _partition_batch_by_arm(self, batch: TransitionBatch) -> List[TransitionBatch]:
        """
        Break input batch down into per-arm batches based on action
        """
        batches = []
        for arm in range(self.n_arms):
            # mask of observations for this arm
            # assume action indices
            mask = batch.action[:, 0] == arm
            if mask.sum().item() == 0:
                # no observations for this arm, use null batch
                batches.append(self._get_null_batch(batch))
            else:
                if batch.state.ndim == 2:
                    # shape: (batch_size, feature_size)
                    # same features for all arms
                    state = batch.state
                elif batch.state.ndim == 3:
                    # shape: (batch_size, num_arms, feature_size)
                    # different features for each arm
                    assert (
                        batch.state.shape[1] == self.n_arms
                    ), "For 3D state, 2nd dimension must be equal to number of arms"
                    state = batch.state[:, arm, :]
                batches.append(
                    TransitionBatch(
                        # pyre-fixme[61]: `state` is undefined, or not always defined.
                        state=state[mask],
                        reward=batch.reward[mask],
                        weight=(
                            batch.weight[mask]
                            if batch.weight is not None
                            else torch.ones_like(mask, dtype=torch.float)
                        ),
                        # empty action features since disjoint model used
                        # action as index of per-arm model
                        # if arms need different features, use 3D `state` instead
                        action=torch.empty(
                            int(mask.sum().item()),
                            0,
                            dtype=torch.float,
                            device=batch.device,
                        ),
                    ).to(batch.device)
                )
        return batches

    def _get_null_batch(self, batch: TransitionBatch) -> TransitionBatch:
        # null batch has 1 element, but 0 weight
        if self._null_batch is None:
            self._null_batch = TransitionBatch(
                state=torch.zeros(
                    1, batch.state.shape[-1], dtype=torch.float, device=batch.device
                ),
                reward=torch.zeros(1, 1, dtype=torch.float, device=batch.device),
                weight=torch.zeros(1, 1, dtype=torch.float, device=batch.device),
                action=torch.empty(
                    1,
                    0,
                    dtype=torch.float,
                    device=batch.device,
                ),
            ).to(batch.device)
        null_batch = self._null_batch
        assert null_batch is not None
        return null_batch

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        action_idx determines which of the models the observation will be routed to.
        """
        self._validate_batch(batch)

        arm_batches = self._partition_batch_by_arm(batch)
        returns = {}
        for i, (arm_bandit, arm_batch) in enumerate(
            zip(self._arm_bandits, arm_batches)
        ):
            returns.update(
                {
                    f"arm_{i}_{k}": v
                    for k, v in arm_bandit.learn_batch(arm_batch).items()
                }
            )
        return returns

    @property
    def models(self) -> List[torch.nn.Module]:
        """
        Get a list of models of each bandit
        """
        return [bandit.model for bandit in self._arm_bandits]

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit: bool = False,
    ) -> Action:
        assert isinstance(available_action_space, DiscreteActionSpace)
        # (batch_size, action_count, feature_size)
        feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=available_action_space,
            state_features_only=self._state_features_only,
            action_representation_module=self._action_representation_module,
        )
        # (batch_size, action_count, feature_size)

        values = ensemble_forward(self.models, feature, use_for_loop=True)
        return self._exploration_module.act(
            subjective_state=feature,
            action_space=available_action_space,
            values=values,
            # pyre-fixme[6]: In call `ExplorationModule.act`, for argument
            # `representation`, expected `Optional[Module]` but got `List[Module]`.
            representation=self.models,
            action_availability_mask=action_availability_mask,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
    ) -> torch.Tensor:
        """
        Returns:
            UCB scores when exploration module is UCB
            Shape is (batch, num_arms) or (num_arms,)
        """
        exploration_module = self._exploration_module
        assert isinstance(exploration_module, ScoreExplorationBase)

        feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            state_features_only=self._state_features_only,
            action_representation_module=self._action_representation_module,
        )
        # (batch_size, action_count, feature_size)

        return exploration_module.get_scores(
            subjective_state=feature,
            values=ensemble_forward(self.models, feature, use_for_loop=True),
            action_space=action_space,
            representation=self.models,  # pyre-fixme[6]: unexpected type
        )

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        # usually, this method would also add the parameters of the history summarization module
        # to the optimizer of the bandit, but disjoint bandits do not use a pytorch optimizer.
        # Instead, the optimization uses Pearl's own linear regression module.
        self._history_summarization_module = value
