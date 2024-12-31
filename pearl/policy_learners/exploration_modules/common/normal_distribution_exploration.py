# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    noise_scaling,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


class NormalDistributionExploration(ExplorationModule):
    """
    Normal Distribution exploration module. Adds gaussian noise to the
    action vector
    """

    def __init__(
        self,
        mean: float = 0.0,
        std_dev: float = 1.0,
    ) -> None:
        super().__init__()
        self._mean = mean
        self._std_dev = std_dev

    def act(
        self,
        action_space: ActionSpace,
        subjective_state: SubjectiveState | None = None,
        values: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        assert isinstance(action_space, BoxActionSpace)
        assert exploit_action is not None
        device = exploit_action.device
        # checks that the exploit action is feasible in the available action space
        low = torch.tensor(action_space.low).to(device)
        high = torch.tensor(action_space.high).to(device)
        assert torch.all(exploit_action >= low) and torch.all(exploit_action <= high)

        action_dim = exploit_action.size()  # dimension of the action space

        # after the design of the action space is set, switch to the below line
        # action_dim = available_action_space.shape[0]  # dimension of the action space

        # generate noise from a standard normal distribution
        noise = torch.normal(
            mean=self._mean,
            std=self._std_dev,
            size=action_dim,
            device=device,
        )

        # scale noise according to the action space
        scaled_noise = noise_scaling(action_space, noise)
        action = exploit_action + scaled_noise  # add noise

        # clip final action value to be within bounds of the action space
        return torch.clamp(action, low, high)

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two NormalDistributionExploration instances for equality,
        checking attributes.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, NormalDistributionExploration):
            differences.append(
                "other is not an instance of NormalDistributionExploration"
            )
        else:
            if self._mean != other._mean:
                differences.append(f"_mean is different: {self._mean} vs {other._mean}")
            if self._std_dev != other._std_dev:
                differences.append(
                    f"_std_dev is different: {self._std_dev} vs {other._std_dev}"
                )

        return "\n".join(differences)
