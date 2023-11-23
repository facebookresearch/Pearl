from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition


class FIFOOnPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int) -> None:
        super(FIFOOnPolicyReplayBuffer, self).__init__(capacity)
        # this is used to delay push SARS
        # wait for next action is available and then final push
        # this is designed for single transition for now
        # pyre-fixme[4]: Attribute must be annotated.
        self.cache = None

    # pyre-fixme[14]: `push` overrides method defined in `ReplayBuffer` inconsistently.
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
        cost: Optional[float] = None,
    ) -> None:
        (
            curr_available_actions_tensor_with_padding,
            curr_available_actions_mask,
        ) = self._create_action_tensor_and_mask(action_space, curr_available_actions)

        (
            next_available_actions_tensor_with_padding,
            next_available_actions_mask,
        ) = self._create_action_tensor_and_mask(action_space, next_available_actions)

        current_state = self._process_single_state(state)
        current_action = self._process_single_action(action, action_space)

        find_match = self.cache is not None and torch.equal(
            self.cache.next_state, current_state
        )
        if find_match:
            # push a complete SARSA into memory
            self.memory.append(
                Transition(
                    state=self.cache.state,
                    action=self.cache.action,
                    reward=self.cache.reward,
                    next_state=self.cache.next_state,
                    next_action=current_action,
                    curr_available_actions=self.cache.curr_available_actions,
                    curr_available_actions_mask=self.cache.curr_available_actions_mask,
                    next_available_actions=self.cache.next_available_actions,
                    next_available_actions_mask=self.cache.next_available_actions_mask,
                    done=self.cache.done,
                ).to(self.device)
            )
        if not done:
            # save current push into cache
            self.cache = Transition(
                state=current_state,
                action=current_action,
                reward=self._process_single_reward(reward),
                next_state=self._process_single_state(next_state),
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_available_actions_mask=curr_available_actions_mask,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_available_actions_mask=next_available_actions_mask,
                done=self._process_single_done(done),
            ).to(self.device)
        else:
            # for terminal state, push directly
            self.memory.append(
                Transition(
                    state=current_state,
                    action=current_action,
                    reward=self._process_single_reward(reward),
                    next_state=self._process_single_state(next_state),
                    # this value doesnt matter, use current_action for same shape
                    next_action=current_action,
                    curr_available_actions=curr_available_actions_tensor_with_padding,
                    curr_available_actions_mask=curr_available_actions_mask,
                    next_available_actions=next_available_actions_tensor_with_padding,
                    next_available_actions_mask=next_available_actions_mask,
                    done=self._process_single_done(done),
                ).to(self.device)
            )
