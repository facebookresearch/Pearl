import torch

from pearl.policy_learners.deep_td_learning import DeepTDLearning
from pearl.replay_buffer.transition import TransitionBatch


class DeepSARSA(DeepTDLearning):
    """
    A Deep Temporal Difference learning policy learner.
    """

    @torch.no_grad()
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.tensor:
        """
        For SARSA, next state values comes from committed next action + next state value
        """
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_action_batch = batch.next_action  # (batch_size x action_dim)

        next_state_action_batch = torch.cat(
            [next_state_batch, next_action_batch], dim=1
        )  # (batch_size x (state_dim + action_dim))
        next_state_action_values = self._Q_target(next_state_action_batch).view(
            (batch_size,)
        )  # (batch_size)

        return next_state_action_values
