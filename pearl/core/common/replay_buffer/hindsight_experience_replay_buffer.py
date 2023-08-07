from typing import Any, Callable

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState

from pearl.core.common.replay_buffer.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)


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
    goal_dim: dimension of goal of the problem. as input to push API will be the final state representation
              so, it is already a cat among eg position in sparse reward environment + goal
              we could need this info in order to split alternative goal after episode terminates
    reward_fn: is the F here: F(state+goal, action) = reward
    done_fn: This is different from paper. Original paper doesnt have it. We need it for games which may end earlier
              If this is not defined, then use done value from original trajectory.
    """

    def __init__(
        self,
        capacity: int,
        goal_dim: int,
        reward_fn: Callable[[Any, Any], float],
        done_fn: Callable[[Any, Any], bool] = None,
    ) -> None:
        super(HindsightExperienceReplayBuffer, self).__init__(capacity=capacity)
        self._goal_dim = goal_dim
        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._trajectory = []  # a list of transition

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: float,
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        action_space: ActionSpace,
        done: bool,
    ) -> None:
        # assuming state and goal are all list, so we could use + to cat
        super(HindsightExperienceReplayBuffer, self).push(
            # input here already have state and goal cat together
            state,
            action,
            reward,
            next_state,
            curr_available_actions,
            next_available_actions,
            action_space,
            done,
        )
        self._trajectory.append(
            (
                state,
                action,
                next_state,
                curr_available_actions,
                next_available_actions,
                action_space,
                done,
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
                action_space,
                done,
            ) in self._trajectory:
                # replace current_goal with additional_goal
                state[-self._goal_dim :] = additional_goal
                next_state[-self._goal_dim :] = additional_goal
                super(HindsightExperienceReplayBuffer, self).push(
                    state,
                    action,
                    self._reward_fn(state, action),
                    next_state,
                    curr_available_actions,
                    next_available_actions,
                    action_space,
                    done if self._done_fn is None else self._done_fn(state, action),
                )
            self._trajectory = []
