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
from pearl.api.history import History
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
    SubjectiveState,
)


class IdentityHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that simply uses the original observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: History = None

    def summarize_history(
        self, observation: Observation, action: Action | None
    ) -> SubjectiveState:
        self.history = observation
        # pyre-fixme[7]: incompatible return type
        # Due to currently incorrect assumption that SubjectiveState
        # is always a Tensor (not the case for tabular Q-learning, for example)
        return observation

    def get_history(self) -> History:
        return self.history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reset(self) -> None:
        self.history = None

    def compare(self, other: HistorySummarizationModule) -> str:
        """
        Compares two IdentityHistorySummarizationModule instances for equality.

        Since this module has no attributes or buffers to compare,
        it only checks if the `other` object is an instance of the same class.

        Args:
        other: The other HistorySummarizationModule to compare with.

        Returns:
        str: A string describing any differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, IdentityHistorySummarizationModule):
            differences.append(
                "other is not an instance of IdentityHistorySummarizationModule"
            )

        return "\n".join(differences)  # Join the differences with newlines
