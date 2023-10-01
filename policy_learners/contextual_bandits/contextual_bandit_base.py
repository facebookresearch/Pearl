from abc import abstractmethod

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch


class ContextualBanditBase(PolicyLearner):
    """
    A base class for Contextual Bandit policy learner.
    """

    def __init__(
        self,
        feature_dim: int,
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(ContextualBanditBase, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            on_policy=False,
            is_action_continuous=False,  # TODO change this in children classes when we add CB for continuous
        )
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @abstractmethod
    # pyre-fixme[15]: `learn_batch` overrides method defined in `PolicyLearner`
    #  inconsistently.
    def learn_batch(self, batch: TransitionBatch) -> None:
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        pass

    @abstractmethod
    # pyre-fixme[3]: Return type must be annotated.
    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ):
        """
        Returns:
            Return scores trained by this contextual bandit algorithm
        """
        pass
