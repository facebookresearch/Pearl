from pearl.api.action_space import ActionSpace
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffer.transition import TransitionBatch


class ContextualBanditBase(PolicyLearner):
    """
    A base class for Contextual Bandit policy learner.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(ContextualBanditBase, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        self._action_space = action_space
        self._state_dim = state_dim

    def learn_batch(self, batch: TransitionBatch) -> None:
        # This needs to be implemented based on what kinds of CB
        raise NotImplementedError("learn_batch is not implemented")
