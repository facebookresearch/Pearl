# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import torch

from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers import TensorBasedReplayBuffer, Transition, TransitionBatch
from pearl.utils import get_subdataclass_specific_attributes
from torch import Tensor


def create_attribute_column_tensor(
    attr_name: str,
    transitions: list[Transition],
) -> torch.Tensor | None:
    """
    Creates a tensor column from a list of transition objects.

    Args:
        attr_name (str): The name of the attribute to extract from each transition.
        transitions (List[Transition]): A list of transition objects.

    Returns:
        Optional[torch.Tensor]: A tensor column containing the extracted attributes.
        If any attribute is None, returns None.
    """
    list_of_values_of_attr = []
    for transition in transitions:
        value_of_attr = getattr(transition, attr_name)
        if value_of_attr is None:
            return None
        list_of_values_of_attr.append(value_of_attr)
    attr_column_tensor = torch.cat(list_of_values_of_attr)
    return attr_column_tensor


def make_replay_buffer_class_for_specific_transition_types(
    TransitionType: type[Transition], TransitionBatchType: type[TransitionBatch]
) -> type[TensorBasedReplayBuffer]:
    """
    Creates a subclass of TensorBasedReplayBuffer with the specified
    TransitionType and TransitionBatchType.

    This is similar to a generic class depending on types TransitionType and TransitionBatchType,
    but due to technicalities, it does not seem possible to implement it as a generic class.
    """

    # We define a local class using the given transition types,
    # and that will be returned as the result.
    class ReplayBufferForGivenTransitionTypes(TensorBasedReplayBuffer):
        # This statement is one reason why making this a generic class does not work;
        # if this is a generic class on TransitionType, then this function call passes
        # the TypeVar, rather than the value of the TypeVar, as an argument,
        # which is not what we want.
        attr_names: list[str] = get_subdataclass_specific_attributes(TransitionType)

        def __init__(
            self,
            capacity: int,
        ) -> None:
            super().__init__(capacity)

        def _store_transition(
            self,
            state: SubjectiveState,
            action: Action,
            reward: Reward,
            terminated: bool,
            truncated: bool,
            curr_available_actions_tensor_with_padding: Tensor | None,
            curr_unavailable_actions_mask: Tensor | None,
            next_state: SubjectiveState | None,
            next_available_actions_tensor_with_padding: Tensor | None,
            next_unavailable_actions_mask: Tensor | None,
            cost: float | None = None,
        ) -> None:
            # Another point that prevents this from being a generic class;
            # in a generic class, TranstionType would be a TypeVar and non-callable.
            self.memory.append(
                TransitionType(
                    state=self._process_non_optional_single_state(state),
                    action=self._process_single_action(action),
                    reward=self._process_single_reward(reward),
                    next_state=self._process_single_state(next_state),
                    curr_available_actions=curr_available_actions_tensor_with_padding,
                    curr_unavailable_actions_mask=curr_unavailable_actions_mask,
                    next_available_actions=next_available_actions_tensor_with_padding,
                    next_unavailable_actions_mask=next_unavailable_actions_mask,
                    terminated=self._process_single_terminated(terminated),
                    truncated=self._process_single_truncated(truncated),
                )
            )

        @staticmethod
        def include_attrs_in_batch(
            attr_names: list[str],
            transitions: list[Transition],
            transition_batch: TransitionBatch,
        ) -> TransitionBatchType:
            new_columns = {
                attr_name: create_attribute_column_tensor(attr_name, transitions)
                for attr_name in attr_names
            }

            transition_batch = TransitionBatchType.from_parent(
                transition_batch, **new_columns
            )

            return transition_batch

        def _create_transition_batch(
            self,
            transitions: list[Transition],
            is_action_continuous: bool,
        ) -> TransitionBatchType:
            transition_batch = super()._create_transition_batch(
                transitions,
                is_action_continuous,
            )

            transition_batch = self.include_attrs_in_batch(
                self.attr_names, transitions, transition_batch
            )

            return transition_batch.to(self.device_for_batches)

    return ReplayBufferForGivenTransitionTypes
