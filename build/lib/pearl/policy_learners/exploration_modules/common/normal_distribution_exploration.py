# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

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
        super(NormalDistributionExploration, self).__init__()
        self._mean = mean
        self._std_dev = std_dev

    def act(
        self,
        action_space: ActionSpace,
        subjective_state: Optional[SubjectiveState] = None,
        values: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
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
        # clip noise to be between [-1, 1]^{action_dim}
        clipped_noise = torch.clip(noise, -1, 1)

        # scale noise according to the action space
        scaled_noise = noise_scaling(action_space, clipped_noise)
        action = exploit_action + scaled_noise  # add noise

        # clip final action value to be within bounds of the action space
        return torch.clamp(action, low, high)
