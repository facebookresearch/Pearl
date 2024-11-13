# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, Iterable, Tuple

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
        super(TabularQLearning, self).__init__(
            exploration_module=EGreedyExploration(exploration_rate),
            on_policy=False,
            is_action_continuous=False,
            requires_tensors=False,  # temporary solution before abstract interfaces
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values: Dict[Tuple[SubjectiveState, int], Value] = {}
        self.debug: bool = debug

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space
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

        return self._exploration_module.act(
            subjective_state,
            available_action_space,
            exploit_action,
        )

    def learn(
        self,
        replay_buffer: ReplayBuffer,
    ) -> Dict[str, Any]:

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
            _max_number_actions,
            _cost,
        ) = transition
        old_q_value = self.q_values.get((state, action.item()), 0)
        next_q_values = [
            self.q_values.get((next_state, next_action.item()), 0)
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
            self.print_debug_information(state, action, reward, next_state, terminated)

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminated": terminated,
        }

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        raise Exception("tabular_q_learning doesnt need learn_batch")

    def print_debug_information(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState,
        terminated: bool,
    ) -> None:
        print("state:", state)
        print("action:", action)
        print("reward:", reward)
        print("next state:", next_state)
        print("terminated:", terminated)
        print("q-values:", self.q_values)

    def __str__(self) -> str:
        exploration_module = self._exploration_module
        assert isinstance(exploration_module, EGreedyExploration)
        items = [
            "α=" + str(self.learning_rate),
            "ε=" + str(exploration_module.epsilon),
            "λ=" + str(self.discount_factor),
        ]
        return "Q-Learning" + (
            " (" + ", ".join(str(item) for item in items) + ")" if items else ""
        )
