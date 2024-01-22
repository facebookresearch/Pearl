# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition


class OnPolicyEpisodicReplayBuffer(TensorBasedReplayBuffer):
    def __init__(
        self,
        capacity: int,
        discounted_factor: float = 1.0,
        has_cost_available: bool = False,
    ) -> None:
        super(OnPolicyEpisodicReplayBuffer, self).__init__(
            capacity=capacity,
            has_next_state=True,
            has_next_action=False,
            has_next_available_actions=False,
            has_cost_available=has_cost_available,
        )
        # this is used to delay push SARS
        # wait for next action is available and then final push
        # this is designed for single transition for now
        self.state_action_cache: List[Transition] = []
        self._discounted_factor = discounted_factor

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        done: bool,
        max_number_actions: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        (
            curr_available_actions_tensor_with_padding,
            curr_unavailable_actions_mask,
        ) = self._create_action_tensor_and_mask(
            max_number_actions, curr_available_actions
        )

        current_state = self._process_single_state(state)
        current_action = self._process_single_action(action)
        next_reward = self._process_single_reward(reward)
        n_state = self._process_single_state(next_state)
        self.state_action_cache.append(
            Transition(
                state=current_state,
                action=current_action,
                reward=next_reward,
                cum_reward=None,
                next_state=n_state,
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                next_available_actions=None,
                next_unavailable_actions_mask=None,
                done=self._process_single_done(done),
            ).to(self.device)
        )

        if done:
            # discounted_return at time i = sum of (self._discounted_factor^(j-i) * Rj) j is [i, T]
            discounted_return = 0.0
            for i in range(len(self.state_action_cache) - 1, -1, -1):
                cum_reward = self.state_action_cache[i].reward + discounted_return
                self.state_action_cache[i].cum_reward = self._process_single_reward(
                    cum_reward
                )
                self.memory.append(self.state_action_cache[i])
                discounted_return = self._discounted_factor * cum_reward

            self.state_action_cache = []
