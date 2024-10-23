# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from dataclasses import dataclass

from typing import List, Optional, Type

import torch

from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.python_utils import get_subclass_specific_attributes
from torch import Tensor


def create_attribute_column_tensor(
    attr_name: str,
    transitions: List[Transition],
) -> Optional[torch.Tensor]:
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


def make_replay_buffer_class(
    TransitionType: Type[Transition], TransitionBatchType: Type[TransitionBatch]
) -> Type[TensorBasedReplayBuffer]:
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
        attr_names: List[str] = get_subclass_specific_attributes(TransitionType)

        def __init__(
            self,
            capacity: int,
            has_cost_available: bool = False,
        ) -> None:
            super().__init__(
                capacity=capacity,
                has_next_state=True,
                has_next_action=False,
                has_next_available_actions=False,
                has_cost_available=has_cost_available,
            )

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
                )
            )

        @staticmethod
        def include_attrs_in_batch(
            attr_names: List[str],
            transitions: List[Transition],
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
            transitions: List[Transition],
            has_next_state: bool,
            has_next_action: bool,
            is_action_continuous: bool,
            has_next_available_actions: bool,
            has_cost_available: bool,
        ) -> TransitionBatchType:
            transition_batch = super()._create_transition_batch(
                transitions,
                has_next_state,
                has_next_action,
                is_action_continuous,
                has_next_available_actions,
                has_cost_available,
            )

            transition_batch = self.include_attrs_in_batch(
                self.attr_names, transitions, transition_batch
            )

            return transition_batch.to(self.device_for_batches)

    return ReplayBufferForGivenTransitionTypes


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
        # Extract attributes from parent_obj using __dict__ and create a new child object
        child_obj = cls(
            **parent_obj.__dict__,
            gae=gae,
            lam_return=lam_return,
            action_probs=action_probs,
            cum_reward=cum_reward,
        )
        return child_obj


OnPolicyReplayBuffer: Type[TensorBasedReplayBuffer] = make_replay_buffer_class(
    OnPolicyTransition, OnPolicyTransitionBatch
)
