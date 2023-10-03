from typing import Optional

import torch

from pearl.api.action_space import ActionSpace
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.deep_td_learning import (
    DeepTDLearning,
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
        exploration_module: Optional[ExplorationModule] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(DeepSARSA, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            exploration_module=exploration_module
            if exploration_module is not None
            else EGreedyExploration(0.05),
            on_policy=True,
            **kwargs,
        )

    @torch.no_grad()
    def _get_next_state_values(
        self,
        batch: TransitionBatch,
        batch_size: int
        # pyre-fixme[11]: Annotation `tensor` is not defined as a type.
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
