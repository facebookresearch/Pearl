from typing import Iterable, Optional

import torch
from pearl.api.action_space import ActionSpace
from pearl.core.common.neural_networks.value_networks import (
    StateActionValueNetworkType,
    VanillaStateActionValueNetwork,
)
from pearl.core.common.policy_learners.exploration_module.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.sequential_decision_making.policy_learners.deep_td_learning import (
    DeepTDLearning,
)


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
        soft_update_tau: float = 1.0,  # no soft update
        network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
        double: bool = False,
        state_output_dim=None,
        action_output_dim=None,
        state_hidden_dims=None,
        action_hidden_dims=None,
    ) -> None:
        super(DeepQLearning, self).__init__(
            exploration_module=exploration_module
            if exploration_module is not None
            else EGreedyExploration(0.05),
            on_policy=False,
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            soft_update_tau=soft_update_tau,
            network_type=network_type,
            state_output_dim=state_output_dim,
            action_output_dim=action_output_dim,
            state_hidden_dims=state_hidden_dims,
            action_hidden_dims=action_hidden_dims,
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

        # for duelling, this does a forward pass; since the batch of next available actions is already input
        next_state_action_values = self._Q_target.get_batch_action_value(
            next_state_batch_repeated, next_available_actions_batch
        ).view(
            batch_size, -1
        )  # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_available_actions_mask_batch] = -float("inf")

        # if double DQN, choose action from _Q with argmax, but use value from _Q_target
        # if non-double DQN, choose action from _Q_target with argmax, as well as value
        if self._double:
            next_state_action_values_Q = self._Q.get_batch_action_value(
                next_state_batch_repeated, next_available_actions_batch
            ).view(
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
