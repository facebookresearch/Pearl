from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition


class FIFOOffPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int) -> None:
        super(FIFOOffPolicyReplayBuffer, self).__init__(
            capacity=capacity, has_next_state=True, has_next_action=False
        )

    # TODO: add helper to convert subjective state into tensors
    # TODO: assumes action space is gym action space with one-hot encoding
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

        (
            curr_available_actions_tensor_with_padding,
            curr_available_actions_mask,
        ) = self._create_action_tensor_and_mask(action_space, curr_available_actions)

        (
            next_available_actions_tensor_with_padding,
            next_available_actions_mask,
        ) = self._create_action_tensor_and_mask(action_space, next_available_actions)

        self.memory.append(
            Transition(
                state=self._process_single_state(state),
                action=self._process_single_action(action, action_space),
                reward=self._process_single_reward(reward),
                next_state=self._process_single_state(next_state),
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_available_actions_mask=curr_available_actions_mask,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_available_actions_mask=next_available_actions_mask,
                done=self._process_single_done(done),
            )
        )
