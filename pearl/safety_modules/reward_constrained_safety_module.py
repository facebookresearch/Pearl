# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule


class RewardConstrainedSafetyModule(SafetyModule):
    """
    Placeholder: to be implemented
    """

    def __init__(self) -> None:
        super(RewardConstrainedSafetyModule, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self._action_space = None

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def filter_action(self, subjective_state: SubjectiveState) -> ActionSpace:
        return self._action_space

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass
