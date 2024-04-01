# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from dataclasses import dataclass

from typing import List, Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch


@dataclass(frozen=False)
class OnPolicyTransition(Transition):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_probs: Optional[torch.Tensor] = None  # action probs
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward


@dataclass(frozen=False)
class OnPolicyTransitionBatch(TransitionBatch):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_probs: Optional[torch.Tensor] = None  # action probs
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward

    @classmethod
    def from_parent(
        cls,
        parent_obj: TransitionBatch,
        gae: Optional[torch.Tensor] = None,
        lam_return: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None,
        cum_reward: Optional[torch.Tensor] = None,
    ) -> "OnPolicyTransitionBatch":
        # Extract attributes from parent_obj using __dict__ and create a new Child object
        child_obj = cls(
            **parent_obj.__dict__,
            gae=gae,
            lam_return=lam_return,
            action_probs=action_probs,
            cum_reward=cum_reward
        )
        return child_obj


class OnPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(
        self,
        capacity: int,
        has_cost_available: bool = False,
    ) -> None:
        super(OnPolicyReplayBuffer, self).__init__(
            capacity=capacity,
            has_next_state=True,
            has_next_action=False,
            has_next_available_actions=False,
            has_cost_available=has_cost_available,
        )

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        terminated: bool,
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
        self.memory.append(
            OnPolicyTransition(
                state=current_state,
                action=current_action,
                reward=next_reward,
                next_state=n_state,
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                next_available_actions=None,
                next_unavailable_actions_mask=None,
                terminated=self._process_single_terminated(terminated),
            ).to(self.device)
        )

    def _create_transition_batch(
        self,
        transitions: List[Transition],
        has_next_state: bool,
        has_next_action: bool,
        is_action_continuous: bool,
        has_next_available_actions: bool,
        has_cost_available: bool,
    ) -> OnPolicyTransitionBatch:
        transition_batch = super()._create_transition_batch(
            transitions,
            has_next_state,
            has_next_action,
            is_action_continuous,
            has_next_available_actions,
            has_cost_available,
        )

        def helper(
            transitions: List[Transition],
            name: str,
        ) -> Optional[torch.Tensor]:
            tmp_list = []
            for x in transitions:
                assert isinstance(x, OnPolicyTransition)
                if getattr(x, name) is None:
                    return None
                tmp_list.append(getattr(x, name))
            return torch.cat(tmp_list)

        names = ["gae", "lam_return", "action_probs", "cum_reward"]
        on_policy_attrs = {}
        for name in names:
            on_policy_attrs[name] = helper(transitions, name)

        transition_batch = OnPolicyTransitionBatch.from_parent(
            transition_batch, **on_policy_attrs
        )

        return transition_batch
