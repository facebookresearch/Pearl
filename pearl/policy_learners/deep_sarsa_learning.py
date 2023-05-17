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


class DeepSARSALearning(DeepTDLearning):
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
    ) -> None:
        super(DeepSARSALearning, self).__init__(
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
        )

    @torch.no_grad()
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.tensor:
        """
        For SARSA, next state values comes from committed next action + next state value
        """
        # TODO we need act_batch here to get next actions
        # implement in later diffs
        raise NotImplementedError("implement act_batch and then continue here")
