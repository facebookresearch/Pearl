# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from pearl.api.action import Action
from pearl.api.history import History
from pearl.api.observation import Observation
from pearl.api.state import SubjectiveState


class HistorySummarizationModule(ABC, nn.Module):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def summarize_history(
        self, observation: Observation, action: Action | None
    ) -> SubjectiveState:
        pass

    @abstractmethod
    def get_history(self) -> History:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def compare(self, other: "HistorySummarizationModule") -> str:
        """
        Compare two history summarization modules and return a string
        describing the differences, or "" if equal.
        """
        pass
