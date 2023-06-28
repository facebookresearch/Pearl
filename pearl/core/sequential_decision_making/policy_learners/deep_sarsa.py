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


class DeepSARSA(DeepTDLearning):
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
        network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
    ) -> None:
        super(DeepSARSA, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            network_type=network_type,
            exploration_module=exploration_module
            if exploration_module is not None
            else EGreedyExploration(0.05),
            on_policy=True,
        )

    @torch.no_grad()
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.tensor:
        """
        For SARSA, next state values comes from committed next action + next state value
        """
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_action_batch = batch.next_action  # (batch_size x action_dim)

        # use get_batch method instead of doing forward pass
        next_state_action_values = self._Q_target.get_batch_action_value(
            next_state_batch, next_action_batch
        )

        return next_state_action_values
