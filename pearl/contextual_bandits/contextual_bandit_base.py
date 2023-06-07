from abc import abstractmethod

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
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

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> None:
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        pass

    @abstractmethod
    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ):
        """
        Returns:
            Return scores trained by this contextual bandit algorithm
        """
        pass
