# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file defines PEARL neural network interafaces
User is free to define their own Q(s, a), but would need to inherit from this interface
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
from torch import nn


class QValueNetwork(abc.ABC, nn.Module):
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
            curr_available_actions_batch (torch.Tensor, optional): a batch of currently available
                actions (batch_size, available_action_space_size, action_dim)
        Returns:
            Q-values of (state, action) pairs: (batch_size)
        """
        ...


class DistributionalQValueNetwork(abc.ABC, nn.Module):
    """
    An interface for estimators of state-action value distribution (Q-value distribution).
    These are value neural networks with special method for computing the Q-value distribution
    and the expected Q-values for a state-action pair.
    Examples include Categorical DQN, Quantile DQN, IQN etc.
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

    @property
    @abc.abstractmethod
    def num_quantiles(self) -> int:
        """Returns number of particles for approximating the quantile distribution"""

    @property
    @abc.abstractmethod
    def quantiles(self) -> torch.Tensor:
        """Returns quantiles of the approximate value distribution"""

    @property
    @abc.abstractmethod
    def quantile_midpoints(self) -> torch.Tensor:
        """Returns midpoints of the quantiles"""

    @abc.abstractmethod
    def get_q_value_distribution(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Returns Z(s, a), a probability distribution over q values, given s and a.
        Note that under a risk neutral measure, Q(s,a) = E[Z(s, a)].
        Args:
            state_batch (torch.Tensor): a batch of state tensors (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of action tensors (batch_size, action_dim)
        Returns:
            approximation of distribution of Q-values of (state, action) pairs
        """
        ...
