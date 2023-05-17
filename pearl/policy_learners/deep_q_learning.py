from typing import Iterable, Optional

import torch
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.factory import NetworkType
from pearl.policy_learners.deep_td_learning import DeepTDLearning
from pearl.policy_learners.exploration_module.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch


class DeepQLearning(DeepTDLearning):
    """
    A Deep Temporal Difference learning policy learner.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_module: Optional[ExplorationModule] = None,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        network_type: NetworkType = NetworkType.VANILLA,
        double: bool = False,
    ) -> None:
        super(DeepQLearning, self).__init__(
            exploration_module=exploration_module
            if exploration_module is not None
            else EGreedyExploration(0.05),
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            network_type=network_type,
        )
        self._double = double

    @torch.no_grad()
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_available_actions_batch = (
            batch.next_available_actions
        )  # (batch_size x action_space_size x action_dim)
        next_available_actions_mask_batch = (
            batch.next_available_actions_mask
        )  # (batch_size x action_space_size)

        next_state_batch_repeated = torch.repeat_interleave(
            next_state_batch.unsqueeze(1), self._action_space.n, dim=1
        )  # (batch_size x action_space_size x state_dim)
        next_state_action_batch = torch.cat(
            [next_state_batch_repeated, next_available_actions_batch], dim=2
        )  # (batch_size x action_space_size x (state_dim + action_dim))
        next_state_action_values = self._Q_target(next_state_action_batch).view(
            (batch_size, -1)
        )  # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_available_actions_mask_batch] = -float("inf")

        # if double DQN, choose action from _Q with argmax, but use value from _Q_target
        # if non-double DQN, choose action from _Q_target with argmax, as well as value
        if self._double:
            next_state_action_values_Q = self._Q(next_state_action_batch).view(
                (batch_size, -1)
            )  # (batch_size x action_space_size)

            # Make sure that unavailable actions' Q values are assigned to -inf
            next_state_action_values_Q[next_available_actions_mask_batch] = -float(
                "inf"
            )
            # Torch.max(1) returns value, indices
            next_state_values_indices = next_state_action_values_Q.max(1)[
                1
            ]  # (batch_size)
            next_state_values = next_state_action_values[
                range(len(next_state_action_values)), next_state_values_indices
            ]
        else:
            # Torch.max(1) returns value, indices
            next_state_values = next_state_action_values.max(1)[0]  # (batch_size)
        return next_state_values
