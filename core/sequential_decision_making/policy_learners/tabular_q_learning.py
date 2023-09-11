import random
from typing import Any, Dict, Optional, Tuple

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Value
from pearl.core.common.policy_learners.exploration_module.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.core.common.policy_learners.policy_learner import PolicyLearner
from pearl.core.common.replay_buffer.replay_buffer import ReplayBuffer
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)

#  TODO make package names and organization more consistent


class TabularQLearning(PolicyLearner):
    """
    A tabular Q-learning policy learner.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.01,
        debug=False,
    ) -> None:
        """
        Initializes the tabular Q-learning policy learner.

        Args:
            learning_rate (float, optional): the learning rate. Defaults to 0.01.
            discount_factor (float, optional): the discount factor. Defaults to 0.9.
            exploration_rate (float, optional): the exploration rate. Defaults to 0.01.
            debug (bool, optional): whether to print debug information to standard output. Defaults to False.
        """
        super(TabularQLearning, self).__init__(
            exploration_module=EGreedyExploration(exploration_rate),
            on_policy=False,
            is_action_continuous=False,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values: Dict[Tuple[SubjectiveState, Action], Value] = {}
        self.debug = debug

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # Choose the action with the highest Q-value for the current state.
        q_values_for_state = {
            action: self.q_values.get((subjective_state, action), 0)
            for action in range(action_space.n)  # TODO: assumes Gym interface, fix it
        }
        max_q_value = max(q_values_for_state.values())
        best_actions = [
            action
            for action, q_value in q_values_for_state.items()
            if q_value == max_q_value
        ]
        exploit_action = random.choice(best_actions)
        if exploit:
            return exploit_action

        return self._exploration_module.act(
            subjective_state,
            action_space,
            exploit_action,
        )

    def learn(
        self,
        replay_buffer: ReplayBuffer,
        _batch_size: Optional[int] = None,
        on_policy: bool = False,
    ) -> None:
        # We currently assume replay buffer only contains last transition (on-policy)
        for (
            state,
            action,
            reward,
            next_state,
            _action_space,
            _curr_available_actions,
            _next_available_actions,
            done,
        ) in replay_buffer.sample(1):

            old_q_value = self.q_values.get((state, action), 0)

            next_q_values = [
                self.q_values.get((next_state, next_action), 0)
                for next_action in range(
                    self._action_space.n
                )  # TODO: assumes Gym interface, fix it
            ]

            if done:
                next_state_value = 0
            else:
                max_next_q_value = max(next_q_values) if next_q_values else 0
                next_state_value = self.discount_factor * max_next_q_value

            new_q_value = old_q_value + self.learning_rate * (
                reward + next_state_value - old_q_value
            )

            self.q_values[(state, action)] = new_q_value

            if self.debug:
                self.print_debug_information(state, action, reward, next_state, done)

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        raise Exception("tabular_q_learning doesnt need learn_batch")

    def print_debug_information(self, state, action, reward, next_state, done):
        print("state:", state)
        print("action:", action)
        print("reward:", reward)
        print("next state:", next_state)
        print("done:", done)
        print("q-values:", self.q_values)

    def __str__(self) -> str:
        items = [
            "α=" + str(self.learning_rate),
            "ε=" + str(self._exploration_module.epsilon),
            "λ=" + str(self.discount_factor),
        ]
        return "Q-Learning" + (
            " (" + ", ".join(str(item) for item in items) + ")" if items else ""
        )
