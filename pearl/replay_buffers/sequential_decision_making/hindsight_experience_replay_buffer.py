# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
    done_fn: This is different from paper. Original paper doesn't have it.
             We need it for games which may end earlier.
             If this is not defined, then use done value from original trajectory.
    """

    # TODO: improve unclear docstring

    def __init__(
        self,
        capacity: int,
        goal_dim: int,
        reward_fn: Callable[[SubjectiveState, Action], Reward],
        done_fn: Optional[Callable[[SubjectiveState, Action], bool]] = None,
    ) -> None:
        super(HindsightExperienceReplayBuffer, self).__init__(capacity=capacity)
        self._goal_dim = goal_dim
        self._reward_fn = reward_fn
        self._done_fn = done_fn
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
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        done: bool,
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
            next_state,
            curr_available_actions,
            next_available_actions,
            done,
            max_number_actions,
            cost,
        )
        self._trajectory.append(
            (
                state,
                action,
                next_state,
                curr_available_actions,
                next_available_actions,
                done,
                max_number_actions,
                cost,
            )
        )
        if done:
            additional_goal = next_state[: -self._goal_dim]  # final mode
            for (
                state,
                action,
                next_state,
                curr_available_actions,
                next_available_actions,
                done,
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
                    next_state,
                    curr_available_actions,
                    next_available_actions,
                    done if self._done_fn is None else self._done_fn(state, action),
                    max_number_actions,
                    cost,
                )
            self._trajectory = []
