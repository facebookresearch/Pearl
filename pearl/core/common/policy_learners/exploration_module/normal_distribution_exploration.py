#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.policy_learners.exploration_module.value_exploration_base import (
    ValueExplorationBase,
)


class NormalDistributionExploration(ValueExplorationBase):
    """
    Normal Distribution exploration module. Add noise to action vector
    """

    def __init__(self, mean: float = 0.0, std_dev: float = 1.0) -> None:
        super(NormalDistributionExploration, self).__init__()
        self._mean = mean
        self._std_dev = std_dev

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        values: torch.Tensor,
        representation: Any,
        exploit_action: Action,
    ) -> Action:
        noise = torch.normal(
            mean=self._mean, std=self._std_dev, size=exploit_action.size()
        )
        return exploit_action + noise
