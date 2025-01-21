# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from collections.abc import Iterable
from typing import Any, List

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward, Value
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.python_utils import first_item
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace

# TODO: make package names and organization more consistent
# TODO: This class currently assumes action index, not generic DiscreteActionSpace.
#   Need to fix this.


class TabularQLearning(PolicyLearner):
    """
    A tabular Q-learning policy learner.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.01,
        debug: bool = False,
    ) -> None:
        """
        Initializes the tabular Q-learning policy learner.
        Currently, tabular Q-learning assumes
        a discrete action space, and assumes that for each action
        int(action.item()) == action's index.

        Args:
            learning_rate (float, optional): the learning rate. Defaults to 0.01.
            discount_factor (float, optional): the discount factor. Defaults to 0.9.
            exploration_rate (float, optional): the exploration rate. Defaults to 0.01.
            debug (bool, optional): whether to print debug information to standard output.
            Defaults to False.
        """
        super().__init__(
            exploration_module=EGreedyExploration(exploration_rate),
            on_policy=False,
            is_action_continuous=False,
            requires_tensors=False,  # temporary solution before abstract interfaces
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values: dict[tuple[SubjectiveState, int], Value] = {}
        self.debug: bool = debug

    def reset(self, action_space: ActionSpace) -> None:
        # pyre-fixme[16]: `TabularQLearning` has no attribute `_action_space`.
        self._action_space = action_space
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        for i, action in enumerate(self._action_space):
            if int(action.item()) != i:
                raise ValueError(
                    f"{self.__class__.__name__} only supports "
                    f"action spaces that are a DiscreteSpace where for each action "
                    f"action.item() == action's index. "
                )

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        # tabular q learning is assumed to not update parameters of the history summarization module
        self._history_summarization_module = value

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        assert isinstance(available_action_space, DiscreteSpace)
        # TODO: if we substitute DiscreteActionSpace for DiscreteSpace
        # we get Pyre errors. It would be nice to fix this.

        # Choose the action with the highest Q-value for the current state.
        action_q_values_for_state = {
            action_index: self.q_values.get((subjective_state, action_index), 0)
            for action_index in range(available_action_space.n)
        }
        max_q_value_for_state = max(action_q_values_for_state.values())
        exploit_action_index = first_item(
            action_index
            for action_index, q_value in action_q_values_for_state.items()
            if q_value == max_q_value_for_state
        )
        exploit_action = available_action_space.actions[exploit_action_index]

        if exploit:
            return exploit_action

        return self.exploration_module.act(
            subjective_state,
            available_action_space,
            exploit_action,
        )

    def learn(
        self,
        replay_buffer: ReplayBuffer,
    ) -> dict[str, Any]:
        # We know the sampling result from SingleTransitionReplayBuffer
        # is a list with a single tuple.
        transitions = replay_buffer.sample(1)
        assert isinstance(transitions, Iterable)
        transition = next(iter(transitions))
        assert isinstance(transition, Iterable)
        # We currently assume replay buffer only contains last transition (on-policy)
        (
            state,
            action,
            reward,
            next_state,
            _curr_available_actions,
            _next_available_actions,
            terminated,
            truncated,
            _max_number_actions,
            _cost,
        ) = transition
        old_q_value = self.q_values.get((state, action.item()), 0)
        next_q_values = [
            self.q_values.get((next_state, next_action.item()), 0)
            # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Tensor, Module]` is not
            #  a function.
            for next_action in self._action_space
        ]

        if terminated:
            next_state_value = 0
        else:
            # pyre-fixme[6]: For 1st argument expected
            #  `Iterable[Variable[SupportsRichComparisonT (bound to
            #  Union[SupportsDunderGT[typing.Any],
            #  SupportsDunderLT[typing.Any]])]]` but got `List[Number]`.
            max_next_q_value = max(next_q_values) if next_q_values else 0
            next_state_value = self.discount_factor * max_next_q_value

        # pyre-fixme[58]: `+` is not supported for operand types `Number` and
        #  `float`.
        # FIXME: not finding a generic assertion that would fix this.
        # assert isinstance(old_q_value, Union[torch.Tensor, int, float])
        # does not work. Pending discussion.
        new_q_value = old_q_value + self.learning_rate * (
            reward + next_state_value - old_q_value
        )

        self.q_values[(state, action.item())] = new_q_value

        if self.debug:
            self.print_debug_information(
                state, action, reward, next_state, terminated, truncated
            )

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminated": terminated,
            "truncated": truncated,
        }

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        raise Exception("tabular_q_learning doesnt need learn_batch")

    def print_debug_information(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState,
        terminated: bool,
        truncated: bool,
    ) -> None:
        print("state:", state)
        print("action:", action)
        print("reward:", reward)
        print("next state:", next_state)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("q-values:", self.q_values)

    def __str__(self) -> str:
        exploration_module = self.exploration_module
        assert isinstance(exploration_module, EGreedyExploration)
        items = [
            "α=" + str(self.learning_rate),
            "ε=" + str(exploration_module.curr_epsilon),
            "λ=" + str(self.discount_factor),
        ]
        return "Q-Learning" + (
            " (" + ", ".join(str(item) for item in items) + ")" if items else ""
        )

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two TabularQLearning instances for equality,
        checking attributes and q-values.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, TabularQLearning):
            differences.append("other is not an instance of TabularQLearning")
        else:  # Type refinement with else block
            # Compare attributes
            if self.learning_rate != other.learning_rate:
                differences.append(
                    f"learning_rate is different: {self.learning_rate} vs {other.learning_rate}"
                )
            if self.discount_factor != other.discount_factor:
                differences.append(
                    f"discount_factor is different: {self.discount_factor} "
                    + f"vs {other.discount_factor}"
                )
            if self.debug != other.debug:
                differences.append(f"debug is different: {self.debug} vs {other.debug}")

            # Compare q-values
            if self.q_values != other.q_values:
                differences.append("q_values are different")

        return "\n".join(differences)

    def get_extra_state(self) -> dict[str, Any]:
        # We must define q_values as extra state since it is
        # not a PyTorch parameter or buffer
        # (which are detected automatically).
        return {
            "q_values": self.q_values,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.q_values = state["q_values"]
