# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import random
from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.tensor_like import assert_is_tensor_like


class DiscreteContextualBanditReplayBuffer(TensorBasedReplayBuffer):
    """
    DiscreteContextualBanditReplayBuffer has the following key differences
    from other replay buffers
    - No next action or next state related
    - action is action idx instead of action value
    - done is not needed, as for contextual bandit, it is always True
    """

    def __init__(self, capacity: int) -> None:
        super(DiscreteContextualBanditReplayBuffer, self).__init__(
            capacity=capacity,
            has_next_state=False,
            has_next_action=False,
            has_next_available_actions=False,
        )

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
        # signature of push is the same as others, in order to match codes in PearlAgent
        # TODO add curr_available_actions and curr_available_actions_mask if needed in the future
        action = assert_is_tensor_like(action)
        self.memory.append(
            Transition(
                state=self._process_single_state(state),
                action=action,
                reward=self._process_single_reward(reward),
            ).to(self.device)
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        samples = random.sample(self.memory, batch_size)
        return TransitionBatch(
            state=torch.cat([x.state for x in samples]),
            action=torch.stack([x.action for x in samples]),
            reward=torch.cat([x.reward for x in samples]),
        ).to(self.device)
