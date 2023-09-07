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

    noise_clip:
        Could be either a constant to apply to all dimensions or a tensor.
        Default to max_action_value / 10.
        TODO we could potentially support dynamic/relative noise clip.
        `max_action_value / 10` can be much larger than `scaled_action`.
        Thus, maybe `scaled_action/10` is another option.
    max_action_value:
        Could be either a constant or a tensor.
        Value comes from what problem user is solving.
        Eg for Pendulum-v1, this is 2.
    min_action_value:
        Same as max action value.
        Default value is -max_action_value.
    """

    def __init__(
        self,
        max_action_value,
        min_action_value,
        mean: float = 0.0,
        std_dev: float = 1.0,
        noise_clip=None,
    ) -> None:
        super(NormalDistributionExploration, self).__init__()
        self._mean = mean
        self._std_dev = std_dev
        self._max_action_value = max_action_value
        self._noise_clip = (
            noise_clip if noise_clip is not None else max_action_value / 10
        )
        self._min_action_value = min_action_value
        assert torch.all(
            torch.tensor(self._noise_clip) >= 0
        ), "clip value for noise should be >= 0"

    def act(
        self,
        exploit_action: Action,
        # arguments below are useless for NormalDistributionExploration
        subjective_state: SubjectiveState = None,
        available_action_space: ActionSpace = None,
        values: torch.Tensor = None,
        representation: Any = None,
    ) -> Action:
        # input exploit_action is from an actor network, which should be normalized to [-1, 1]
        assert torch.all(exploit_action >= -1) and torch.all(exploit_action <= 1)
        # generate noise
        noise = torch.normal(
            mean=self._mean,
            std=self._std_dev,
            size=exploit_action.size(),
            device=self.device,
        )
        # clip noise
        noise = torch.clamp(noise, -self._noise_clip, self._noise_clip)
        # scale normalized exploit_action
        # project from [-1, 1] to [self._min_action_value, self._max_action_value]
        scaled_action = (self._max_action_value - self._min_action_value) * (
            exploit_action + 1
        ) / 2 + self._min_action_value
        # add noise
        scaled_action += noise
        # clip on final action value
        return torch.clamp(
            scaled_action, self._min_action_value, self._max_action_value
        )
