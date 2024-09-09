# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional

from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition
from torch import Tensor


class FIFOOffPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int, has_cost_available: bool = False) -> None:
        super(FIFOOffPolicyReplayBuffer, self).__init__(
            capacity=capacity,
            has_next_state=True,
            has_next_action=False,
            has_cost_available=has_cost_available,
        )

    def _store_transition(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        curr_available_actions_tensor_with_padding: Optional[Tensor],
        curr_unavailable_actions_mask: Optional[Tensor],
        next_state: Optional[SubjectiveState],
        next_available_actions_tensor_with_padding: Optional[Tensor],
        next_unavailable_actions_mask: Optional[Tensor],
        cost: Optional[float] = None,
    ) -> None:
        transition = Transition(
            state=self._process_non_optional_single_state(state),
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
        self.memory.append(transition)
