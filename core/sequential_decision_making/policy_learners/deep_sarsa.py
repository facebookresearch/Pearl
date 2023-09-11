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


class DeepSARSA(DeepTDLearning):
    """
    A Deep Temporal Difference learning policy learner.
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
        network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        network_instance: Optional[QValueNetwork] = None,
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
            network_instance=network_instance,
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
        assert next_state_batch is not None, "SARSA needs to have next state"
        assert next_action_batch is not None, "SARSA needs to have next action"

        # use get_batch method instead of doing forward pass
        next_state_action_values = self._Q_target.get_q_values(
            next_state_batch, next_action_batch
        )

        return next_state_action_values
