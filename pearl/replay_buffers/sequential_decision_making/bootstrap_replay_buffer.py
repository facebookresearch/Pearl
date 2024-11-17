# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random
from typing import Optional

import torch
from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers import BasicReplayBuffer  # noqa E501
from pearl.replay_buffers.transition import (
    TransitionWithBootstrapMask,
    TransitionWithBootstrapMaskBatch,
)
from torch import Tensor


class BootstrapReplayBuffer(BasicReplayBuffer):
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
        # sample the bootstrap mask from Bernoulli(p) on each push
        probs = torch.tensor(self.p).repeat(1, self.ensemble_size)
        bootstrap_mask = torch.bernoulli(probs)

        self.memory.append(
            TransitionWithBootstrapMask(
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
            # pyre-fixme[6]: For 1st argument expected `List[Transition]` but got
            #  `List[Union[Transition, TransitionBatch]]`.
            transitions=samples,
            is_action_continuous=self._is_action_continuous,
        )
        # pyre-fixme[16]: Item `Transition` of `Union[Transition, TransitionBatch]`
        #  has no attribute `bootstrap_mask`.
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
            terminated=transition_batch.terminated,
            bootstrap_mask=bootstrap_mask_batch,
        )
