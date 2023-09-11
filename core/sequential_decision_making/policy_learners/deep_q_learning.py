from typing import Iterable, Optional, Type

import torch
from pearl.api.action_space import ActionSpace
from pearl.core.sequential_decision_making.policy_learners.deep_td_learning import (
    DeepTDLearning,
)
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch


class DeepQLearning(DeepTDLearning):
    """
    Deep Q Learning Policy Learner
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Optional[Iterable[int]] = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_module: Optional[ExplorationModule] = None,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 1.0,  # no soft update
        network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        is_conservative: bool = False,
        conservative_alpha: float = 2.0,
        state_output_dim=None,
        action_output_dim=None,
        state_hidden_dims=None,
        action_hidden_dims=None,
        network_instance: Optional[QValueNetwork] = None,
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
            is_conservative=is_conservative,
            conservative_alpha=conservative_alpha,
            network_type=network_type,
            state_output_dim=state_output_dim,
            action_output_dim=action_output_dim,
            state_hidden_dims=state_hidden_dims,
            action_hidden_dims=action_hidden_dims,
            network_instance=network_instance,
        )

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
        next_state_action_values = self._Q_target.get_q_values(
            next_state_batch_repeated, next_available_actions_batch
        ).view(
            batch_size, -1
        )  # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_available_actions_mask_batch] = -float("inf")

        # Torch.max(1) returns value, indices
        return next_state_action_values.max(1)[0]  # (batch_size)
