# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition


class FIFOOffPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int, has_cost_available: bool = False) -> None:
        super(FIFOOffPolicyReplayBuffer, self).__init__(
            capacity=capacity,
            has_next_state=True,
            has_next_action=False,
            has_cost_available=has_cost_available,
        )

    # TODO: add helper to convert subjective state into tensors
    # TODO: assumes action space is gym action space with one-hot encoding
    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        curr_available_actions: Optional[ActionSpace] = None,
        next_state: Optional[SubjectiveState] = None,
        next_available_actions: Optional[ActionSpace] = None,
        max_number_actions: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        if curr_available_actions is None:
            raise ValueError(
                f"{type(self)} requires curr_available_actions not to be None"
            )

        if next_available_actions is None:
            raise ValueError(
                f"{type(self)} requires next_available_actions not to be None"
            )

        if next_state is None:
            raise ValueError(f"{type(self)} requires next_state not to be None")

        (
            curr_available_actions_tensor_with_padding,
            curr_unavailable_actions_mask,
        ) = self._create_action_tensor_and_mask(
            max_number_actions, curr_available_actions
        )

        (
            next_available_actions_tensor_with_padding,
            next_unavailable_actions_mask,
        ) = self._create_action_tensor_and_mask(
            max_number_actions, next_available_actions
        )
        self.memory.append(
            Transition(
                state=self._process_single_state(state),
                action=self._process_single_action(action),
                reward=self._process_single_reward(reward),
                next_state=self._process_single_state(next_state),
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_unavailable_actions_mask=next_unavailable_actions_mask,
                terminated=self._process_single_terminated(terminated),
                cost=self._process_single_cost(cost),
            )
        )
