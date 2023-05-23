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
        # TODO we need act_batch here to get next actions
        # implement in later diffs
        raise NotImplementedError("implement act_batch and then continue here")
