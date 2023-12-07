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
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (  # noqa E501
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.transition import (
    TransitionWithBootstrapMask,
    TransitionWithBootstrapMaskBatch,
)


class BootstrapReplayBuffer(FIFOOffPolicyReplayBuffer):
    r"""A ensemble replay buffer that supports the implementation of the
    masking distribution used in Bootstrapped DQN, as described in [1]. This
    implementation uses a Bernoulli(p) masking distribution (see Appendix 3.1
    of [1]). The `k`-th Q-network receives an independently drawn mask
    `w_k ~ Bernoulli(p)` for each piece of experience, and `w_k = 1` means
    the experience is included in the training data.

    [1] Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin
        Van Roy, Deep exploration via bootstrapped DQN. Advances in Neural
        Information Processing Systems, 2016. https://arxiv.org/abs/1602.04621.

    Args:
        capacity: Size of the replay buffer.
        p: The parameter of the Bernoulli masking distribution.
        ensemble_size: The number of Q-networks in the ensemble.
        has_next_state: Whether each piece of experience includes the next state.
        has_next_action: Whether each piece of experience includes the next action.
        has_next_available:actions: Whether each piece of experience includes the
            next available actions.
    """

    def __init__(
        self,
        capacity: int,
        p: float,
        ensemble_size: int,
    ) -> None:
        super().__init__(capacity=capacity)
        self.p = p
        self.ensemble_size = ensemble_size

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
        # sample the bootstrap mask from Bernoulli(p) on each push
        probs = torch.tensor(self.p).repeat(1, self.ensemble_size)
        bootstrap_mask = torch.bernoulli(probs)
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
            TransitionWithBootstrapMask(
                state=self._process_single_state(state),
                action=self._process_single_action(action),
                reward=self._process_single_reward(reward),
                next_state=self._process_single_state(next_state),
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_unavailable_actions_mask=next_unavailable_actions_mask,
                done=self._process_single_done(done),
                cost=self._process_single_cost(cost),
                bootstrap_mask=bootstrap_mask,
            )
        )

    def sample(self, batch_size: int) -> TransitionWithBootstrapMaskBatch:
        if batch_size > len(self):
            raise ValueError(
                f"Can't get a batch of size {batch_size} from a "
                f"replay buffer with only {len(self)} elements"
            )
        samples = random.sample(self.memory, batch_size)
        transition_batch = self._create_transition_batch(
            transitions=samples,
            has_next_state=self._has_next_state,
            has_next_action=self._has_next_action,
            is_action_continuous=self.is_action_continuous,
            has_next_available_actions=self._has_next_available_actions,
            has_cost_available=self.has_cost_available,
        )
        bootstrap_mask_batch = torch.cat([x.bootstrap_mask for x in samples])
        return TransitionWithBootstrapMaskBatch(
            state=transition_batch.state,
            action=transition_batch.action,
            reward=transition_batch.reward,
            next_state=transition_batch.next_state,
            curr_available_actions=transition_batch.curr_available_actions,
            curr_unavailable_actions_mask=transition_batch.curr_unavailable_actions_mask,
            next_available_actions=transition_batch.next_available_actions,
            next_unavailable_actions_mask=transition_batch.next_unavailable_actions_mask,
            done=transition_batch.done,
            bootstrap_mask=bootstrap_mask_batch,
        )
