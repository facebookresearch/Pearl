# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import torch
from Pearl.pearl.api.action import Action
from Pearl.pearl.api.history import History
from Pearl.pearl.api.observation import Observation
from Pearl.pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from Pearl.pearl.utils.tensor_like import assert_is_tensor_like


class StackingHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that simply stacks observations into a history.
    """

    def __init__(
        self, observation_dim: int, action_dim: int, history_length: int = 8
    ) -> None:
        super(StackingHistorySummarizationModule, self).__init__()
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.register_buffer("default_action", torch.zeros((1, action_dim)))
        self.register_buffer(
            "history", torch.zeros((history_length, action_dim + observation_dim))
        )

    def summarize_history(
        self, observation: Observation, action: Optional[Action]
    ) -> torch.Tensor:
        if action is None:
            action = self.default_action

        observation = assert_is_tensor_like(observation)
        action = assert_is_tensor_like(action)

        assert observation.shape[-1] + action.shape[-1] == self.history.shape[-1], (
            f"{observation.shape=} {action.shape=} {self.history.shape=}"
        )

        if action.dim() == 1:
            # Then create a tensor of shape (1, action_dim)
            action = action.view(1, -1)

        assert action.dim() == 2, f"{action.dim()=}"

        # The observation takes on a dimension of 2 when we run:
        # observation = observation.view(1, -1)

        observation_action_pair = torch.cat(
            (action, observation.view(1, -1)), dim=-1
        ).detach()

        self.history = torch.cat(
            [
                self.history[1:, :],
                observation_action_pair.view(
                    (1, self.action_dim + self.observation_dim)
                ),
            ],
            dim=0,
        )
        return self.history.view((-1))

    def get_history(self) -> torch.Tensor:
        return self.history.view((-1))

    def forward(self, x: History) -> torch.Tensor:
        try:
            x = assert_is_tensor_like(x)
        except AssertionError as ae:
            raise AssertionError(f"{type(x)=} {x=} is not a tensor like object.") from ae
        return x

    def reset(self) -> None:
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )
