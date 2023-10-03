from typing import Optional, Tuple

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


class DeepQLearning(DeepTDLearning):
    """
    Deep Q Learning Policy Learner
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        learning_rate: float = 0.001,
        exploration_module: Optional[ExplorationModule] = None,
        soft_update_tau: float = 1.0,  # no soft update
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(DeepQLearning, self).__init__(
            exploration_module=exploration_module
            if exploration_module is not None
            else EGreedyExploration(0.05),
            on_policy=False,
            state_dim=state_dim,
            action_space=action_space,
            learning_rate=learning_rate,
            soft_update_tau=soft_update_tau,
            **kwargs,
        )

    @torch.no_grad()
    def _get_next_state_values(
        self,
        batch: TransitionBatch,
        batch_size: int
        # pyre-fixme[11]: Annotation `tensor` is not defined as a type.
    ) -> torch.tensor:
        (
            next_state,
            next_avail_actions,
            next_avail_actions_mask,
        ) = self._prepare_next_state_action_batch(batch)

        # for dueling, this does a forward pass; since the batch of next available
        # actions is already input
        # (batch_size x action_space_size)
        next_state_action_values = self._Q_target.get_q_values(
            next_state, next_avail_actions
        ).view(batch_size, -1)

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_avail_actions_mask] = -float("inf")

        # Torch.max(1) returns value, indices
        return next_state_action_values.max(1)[0]  # (batch_size)

    def _prepare_next_state_action_batch(
        self, batch: TransitionBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        # (batch_size x action_space_size x action_dim)
        next_available_actions_batch = batch.next_available_actions

        # (batch_size x action_space_size)
        next_available_actions_mask_batch = batch.next_available_actions_mask

        # (batch_size x action_space_size x state_dim)
        next_state_batch_repeated = torch.repeat_interleave(
            next_state_batch.unsqueeze(1), self._action_space.n, dim=1  # pyre-ignore
        )
        return (
            next_state_batch_repeated,
            next_available_actions_batch,
            next_available_actions_mask_batch,
        )
