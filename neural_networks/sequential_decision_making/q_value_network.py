#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
This file defines PEARL neural network interafaces
User is free to define their own Q(s, a), but would need to inherit from this interface
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
from pearl.neural_networks.common.auto_device_nn_module import AutoDeviceNNModule


class QValueNetwork(abc.ABC, AutoDeviceNNModule):
    """
    An interface for state-action value (Q-value) estimators (typically, neural networks).
    These are value neural networks with a special method
    for computing the Q-value for a state-action pair.
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Returns state dimention"""
        ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Returns action dimention"""
        ...

    @abc.abstractmethod
    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns Q(s, a), given s and a
        Args:
            state_batch (torch.Tensor): a batch of state tensors (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of action tensors (batch_size, action_dim)
            curr_available_actions_batch (torch.Tensor, optional): a batch of currently available actions (batch_size, available_action_space_size, action_dim)
        Returns:
            Q-values of (state, action) pairs: (batch_size)
        """
        ...
