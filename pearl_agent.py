from typing import Any, Dict

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.agent import Agent
from pearl.api.observation import Observation
from pearl.api.state import SubjectiveState
from pearl.core.common.replay_buffer.single_transition_replay_buffer import (
    SingleTransitionReplayBuffer,
)
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.common.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)


class PearlAgent(Agent):
    """
    A Agent gathering the most common aspects of production-ready agents.
    It is meant as a catch-all agent whose functionality is defined by flags
    (and possibly factories down the line)
    """

    default_safety_module_type = IdentitySafetyModule
    default_history_summarization_module_type = IdentityHistorySummarizationModule
    default_replay_buffer_type = SingleTransitionReplayBuffer

    # TODO: define a data structure that hosts the configs for a Pearl Agent
    def __init__(
        self,
        policy_learner,
        safety_module=None,
        replay_buffer=None,
        history_summarization_module=None,
    ) -> None:
        """
        Initializes the PearlAgent.
        Args:
            policy_learner: a PolicyLearner instance
            safety_module: (optional) a SafetyModule instance (default is IdentitySafetyModule)
            history_summarization_module: (optional) a HistorySummarizationModule instance (default is IdentityHistorySummarizationModule)
            replay_buffer: (optional) a replay buffer (default is single-transition replay buffer for now -- will very likely to change)
        """
        self.policy_learner = policy_learner
        self.safety_module = (
            PearlAgent.default_safety_module_type()
            if safety_module is None
            else safety_module
        )
        self.replay_buffer = (
            PearlAgent.default_replay_buffer_type()
            if replay_buffer is None
            else replay_buffer
        )
        self.history_summarization_module = (
            PearlAgent.default_history_summarization_module_type()
            if history_summarization_module is None
            else history_summarization_module
        )

        # set here so replay_buffer and policy_learner are in sync
        self.replay_buffer.is_action_continuous = (
            self.policy_learner.is_action_continuous
        )

        self._subjective_state = None
        self._latest_action = None

    def act(self, exploit: bool = False) -> Action:
        safe_action_space = self.safety_module.filter_action(self._subjective_state)
        self._latest_action = self.policy_learner.act(
            self._subjective_state, safe_action_space, exploit
        )
        return self._latest_action

    def observe(
        self,
        action_result: ActionResult,
    ) -> None:
        new_subjective_state = self._update_subjective_state(action_result.observation)

        # TODO: define each push with a uuid
        # TODO: currently assumes the same action space across all steps
        # need to modify ActionResults
        self.replay_buffer.push(
            self._subjective_state,
            self._latest_action,
            action_result.reward,
            new_subjective_state,
            self._action_space,  # curr_available_actions
            self._action_space,  # next_available_actions
            self._action_space,  # action_space
            action_result.done,
        )

        self._subjective_state = new_subjective_state

    def learn(self) -> Dict[str, Any]:
        report = self.policy_learner.learn(self.replay_buffer)
        self.safety_module.learn(self.replay_buffer)
        self.history_summarization_module.learn(self.replay_buffer)

        if self.policy_learner.on_policy:
            self.replay_buffer.empty()

        return report

    def learn_batch(self, batch: TransitionBatch) -> None:
        """
        This API is often used in offline learning
        where users pass in a batch of data to train directly
        """
        self.policy_learner.learn_batch(batch)
        self.safety_module.learn_batch(batch)
        self.history_summarization_module.learn_batch(batch)

    def reset(self, observation: Observation, action_space: ActionSpace) -> None:
        self._subjective_state = self._update_subjective_state(observation)
        self._action_space = action_space
        self.safety_module.reset(action_space)
        self.policy_learner.reset(action_space)

    def _update_subjective_state(self, observation: Observation) -> SubjectiveState:
        return self.history_summarization_module.summarize_history(
            self._subjective_state, observation
        )

    def __str__(self) -> str:
        items = []
        items.append(self.policy_learner)
        if type(self.safety_module) is not PearlAgent.default_safety_module_type:
            items.append(self.safety_module)
        if (
            type(self.history_summarization_module)
            is not PearlAgent.default_history_summarization_module_type
        ):
            items.append(self.history_summarization_module)
        if type(self.replay_buffer) is not PearlAgent.default_replay_buffer_type:
            items.append(self.replay_buffer)
        return "PearlAgent" + (
            " with " + ", ".join(str(item) for item in items) if items else ""
        )
