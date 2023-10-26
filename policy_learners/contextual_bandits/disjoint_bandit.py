#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Dict, List, Optional

import torch

from pearl.api.action import Action
from pearl.history_summarization_modules.history_summarization_module import (
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
from pearl.utils.device import get_pearl_device
from pearl.utils.instantiations.action_spaces.action_spaces import (
    ActionSpace,
    DiscreteActionSpace,
)


class DisjointBanditContainer(ContextualBanditBase):
    """
    Wrapper for disjoint models with discrete (and usually small) action space.
    Each action has its own bandit model (can be based on UCB or Thompson Sampling).
    Using the Composite design pattern: https://refactoring.guru/design-patterns/composite
    """

    def __init__(
        self,
        feature_dim: int,
        arm_bandits: List[ContextualBanditBase],
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(DisjointBanditContainer, self).__init__(
            feature_dim=feature_dim,
            exploration_module=exploration_module,
            training_rounds=training_rounds,
            batch_size=batch_size,
        )
        self.device: torch.device = get_pearl_device()
        # Currently our disjoint LinUCB usecase only use LinearRegression
        self._arm_bandits: torch.nn.ModuleList = torch.nn.ModuleList(arm_bandits).to(
            self.device
        )
        self._n_arms: int = len(arm_bandits)

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
            mask = batch.action == arm
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
                    state=state[mask],
                    reward=batch.reward[mask],
                    weight=batch.weight[mask]
                    if batch.weight is not None
                    else torch.ones_like(mask, dtype=torch.float),
                    # empty action features since disjoint model used action as index of per-arm model
                    # if arms need different features, use 3D `state` instead
                    action=torch.empty(
                        int(mask.sum().item()),
                        0,
                        dtype=torch.float,
                        device=self.device,
                    ),
                )
            )
        return batches

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
            if len(arm_batch) == 0:
                # skip updates if batch has no observations for this arm
                continue
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
        # pyre-fixme[16]: `ActionSpace` has no attribute `cat_state_tensor`.
        feature = available_action_space.cat_state_tensor(
            subjective_state=subjective_state
        )  # batch_size, action_count, feature_size

        values = ensemble_forward(self.models, feature)

        return self._exploration_module.act(
            subjective_state=feature,
            action_space=available_action_space,
            values=values,
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
        assert isinstance(self._exploration_module, ScoreExplorationBase)
        feature = action_space.cat_state_tensor(
            subjective_state=subjective_state
        )  # batch_size, action_count, feature_size
        return self._exploration_module.get_scores(
            subjective_state=feature,
            values=ensemble_forward(self.models, feature),
            # when action_space is None, we are querying score for one action
            action_space=action_space
            if action_space is not None
            else DiscreteActionSpace([0]),
            representation=self.models,
        ).squeeze()
