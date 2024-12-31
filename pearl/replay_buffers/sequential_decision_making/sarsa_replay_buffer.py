# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import torch

from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition
from torch import Tensor


class SARSAReplayBuffer(TensorBasedReplayBuffer):
    """
    This replay buffer is used to delay push for SARSA.
    It waits until next action is available and only then does it push a transition
    that contains that information.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.cache: Transition | None = None

    def _store_transition(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        curr_available_actions_tensor_with_padding: Tensor | None,
        curr_unavailable_actions_mask: Tensor | None,
        next_state: SubjectiveState | None,
        next_available_actions_tensor_with_padding: Tensor | None,
        next_unavailable_actions_mask: Tensor | None,
        cost: float | None = None,
    ) -> None:
        current_state = self._process_non_optional_single_state(state)
        current_action = self._process_single_action(action)

        if self.cache is not None:
            assert self.cache.next_state is not None
            find_match = torch.equal(self.cache.next_state, current_state)
        else:
            find_match = False

        if find_match:
            # push a complete SARSA into memory
            assert self.cache is not None
            self.memory.append(
                Transition(
                    state=self.cache.state,
                    action=self.cache.action,
                    reward=self.cache.reward,
                    next_state=self.cache.next_state,
                    next_action=current_action,
                    curr_available_actions=self.cache.curr_available_actions,
                    curr_unavailable_actions_mask=self.cache.curr_unavailable_actions_mask,
                    next_available_actions=self.cache.next_available_actions,
                    next_unavailable_actions_mask=self.cache.next_unavailable_actions_mask,
                    terminated=self.cache.terminated,
                    truncated=self.cache.truncated,
                )
            )
        if not (terminated or truncated):
            # save current push into cache
            self.cache = Transition(
                state=current_state,
                action=current_action,
                reward=self._process_single_reward(reward),
                next_state=self._process_single_state(next_state),
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_unavailable_actions_mask=next_unavailable_actions_mask,
                terminated=self._process_single_terminated(terminated),
                truncated=self._process_single_truncated(truncated),
            )
        else:
            # for terminal state or time out, push directly
            self.memory.append(
                Transition(
                    state=current_state,
                    action=current_action,
                    reward=self._process_single_reward(reward),
                    next_state=self._process_single_state(next_state),
                    # this value doesnt matter, use current_action for same shape
                    next_action=current_action,
                    curr_available_actions=curr_available_actions_tensor_with_padding,
                    curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                    next_available_actions=next_available_actions_tensor_with_padding,
                    next_unavailable_actions_mask=next_unavailable_actions_mask,
                    terminated=self._process_single_terminated(terminated),
                    truncated=self._process_single_truncated(truncated),
                )
            )
