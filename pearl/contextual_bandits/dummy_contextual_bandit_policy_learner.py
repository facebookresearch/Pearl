from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.contextual_bandits.contextual_bandit_base import ContextualBanditBase
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.no_exploration import NoExploration
from pearl.replay_buffer.transition import TransitionBatch


class DummyContextualBanditPolicyLearner(ContextualBanditBase):
    """
    A temporary class showing the structure of a context bandit policy learner.
    """

    def __init__(self) -> None:
        super(DummyContextualBanditPolicyLearner, self).__init__(
            state_dim=None,  # dummy
            action_space=None,  # dummy
            exploration_module=NoExploration(),
        )

    def exploit(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
    ) -> Action:
        # Code making the decision
        # SubjectiveState will be the same type as the Observation coming out of the environment
        # if no history summarization module is being used.
        # If such a module is being used, the SubjectiveState will whatever type that module provides.
        action = 0
        return action

    def learn_batch(self, batch: TransitionBatch) -> None:
        # Code doing the learning for the provided data
        pass
