# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Callable, List, Optional, Tuple

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.tensor_like import assert_is_tensor_like


class HindsightExperienceReplayBuffer(FIFOOffPolicyReplayBuffer):
    """
    paper: https://arxiv.org/pdf/1707.01495.pdf
    final mode for alternative only for now

    TLDR:
    HindsightExperienceReplayBuffer is used for sparse reward problems.
    After an episode ends, apart from pushing original data in,
    it will replace original goal with final state in the episode,
    and replay the transitions again for new rewards and push

    capacity: size of the replay buffer
    goal_dim: dimension of goal of the problem.
              Subjective state input to `push` method will be the final state representation
              so we could need this info in order to split alternative goal after episode
              terminates.
    reward_fn: is the F here: F(state+goal, action) = reward
    terminated_fn: This is different from paper. Original paper doesn't have it.
             We need it for games which may end earlier.
             If this is not defined, then use terminated value from original trajectory.
    """

    # TODO: improve unclear docstring

    def __init__(
        self,
        capacity: int,
        goal_dim: int,
        reward_fn: Callable[[SubjectiveState, Action], Reward],
        terminated_fn: Optional[Callable[[SubjectiveState, Action], bool]] = None,
    ) -> None:
        super(HindsightExperienceReplayBuffer, self).__init__(capacity=capacity)
        self._goal_dim = goal_dim
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._trajectory: List[
            Tuple[
                SubjectiveState,
                Action,
                SubjectiveState,
                ActionSpace,
                ActionSpace,
                bool,
                Optional[int],
                Optional[float],
            ]
        ] = []

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
        next_state = assert_is_tensor_like(next_state)
        # assuming state and goal are all list, so we could use + to cat
        super(HindsightExperienceReplayBuffer, self).push(
            # input here already have state and goal cat together
            state,
            action,
            reward,
            terminated,
            curr_available_actions,
            next_state,
            next_available_actions,
            max_number_actions,
            cost,
        )

        if curr_available_actions is None:
            raise ValueError(
                f"{type(self)} requires curr_available_actions not to be None"
            )

        if next_available_actions is None:
            raise ValueError(
                f"{type(self)} requires next_available_actions not to be None"
            )

        self._trajectory.append(
            (
                state,
                action,
                next_state,
                curr_available_actions,
                next_available_actions,
                terminated,
                max_number_actions,
                cost,
            )
        )
        if terminated:
            additional_goal = next_state[: -self._goal_dim]  # final mode
            for (
                state,
                action,
                next_state,
                curr_available_actions,
                next_available_actions,
                terminated,
                max_number_actions,
                cost,
            ) in self._trajectory:
                # replace current_goal with additional_goal
                state = assert_is_tensor_like(state)
                next_state = assert_is_tensor_like(next_state)
                state[-self._goal_dim :] = additional_goal
                next_state[-self._goal_dim :] = additional_goal
                super(HindsightExperienceReplayBuffer, self).push(
                    state,
                    action,
                    self._reward_fn(state, action),
                    (
                        terminated
                        if self._terminated_fn is None
                        else self._terminated_fn(state, action)
                    ),
                    curr_available_actions,
                    next_state,
                    next_available_actions,
                    max_number_actions,
                    cost,
                )
            self._trajectory = []
