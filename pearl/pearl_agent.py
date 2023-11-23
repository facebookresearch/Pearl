from typing import Any, Dict, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.agent import Agent
from pearl.api.observation import Observation
from pearl.api.state import SubjectiveState
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.policy_learners.policy_learner import (
    DistributionalPolicyLearner,
    PolicyLearner,
)
from pearl.replay_buffers.examples.single_transition_replay_buffer import (
    SingleTransitionReplayBuffer,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.safety_modules.risk_sensitive_safety_modules import RiskNeutralSafetyModule
from pearl.safety_modules.safety_module import SafetyModule
from pearl.utils.compatibility_checks import pearl_agent_compatibility_check
from pearl.utils.device import get_pearl_device
from pearl.utils.instantiations.action_spaces.discrete import DiscreteActionSpace


class PearlAgent(Agent):
    """
    A Agent gathering the most common aspects of production-ready agents.
    It is meant as a catch-all agent whose functionality is defined by flags
    (and possibly factories down the line)
    """

    default_safety_module_type = IdentitySafetyModule
    default_risk_sensitive_safety_module_type = RiskNeutralSafetyModule
    default_history_summarization_module_type = IdentityHistorySummarizationModule
    default_replay_buffer_type = SingleTransitionReplayBuffer
    default_action_representation_module_type = IdentityActionRepresentationModule

    # TODO: define a data structure that hosts the configs for a Pearl Agent
    def __init__(
        self,
        policy_learner: PolicyLearner,
        safety_module: Optional[SafetyModule] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        history_summarization_module: Optional[HistorySummarizationModule] = None,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        device_id: int = -1,
    ) -> None:
        """
        Initializes the PearlAgent.
        Args:
            policy_learner: a PolicyLearner instance
            safety_module: (optional) a SafetyModule instance (default is
                    RiskNeutralSafetyModule for distributional policy learner
                    types and IdentitySafetyModule for all other types)
            risk_sensitive_safety_module: (optional) a RiskSensitiveSafetyModule
                    instance (default is RiskNeutralSafetyModule)
            history_summarization_module: (optional) a HistorySummarizationModule
                    instance (default is IdentityHistorySummarizationModule)
            replay_buffer: (optional) a replay buffer (default is single-transition
                    replay buffer for now -- will very likely change)
        """
        self.policy_learner: PolicyLearner = policy_learner
        self._device_id: int = device_id
        self.device: torch.device = get_pearl_device(device_id)

        # pyre-fixme[4]: Attribute must be annotated.
        self.safety_module = (
            safety_module
            if safety_module is not None
            else (
                PearlAgent.default_risk_sensitive_safety_module_type()
                if isinstance(self.policy_learner, (DistributionalPolicyLearner))
                else PearlAgent.default_safety_module_type()
            )
        )

        # adds the safety module to the policy learner as well
        # @jalaj, we need to follow the practice below for safety module
        self.policy_learner.safety_module = self.safety_module

        # pyre-fixme[4]: Attribute must be annotated.
        self.replay_buffer = (
            PearlAgent.default_replay_buffer_type()
            if replay_buffer is None
            else replay_buffer
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.history_summarization_module = (
            PearlAgent.default_history_summarization_module_type()
            if history_summarization_module is None
            else history_summarization_module
        )

        self.action_representation_module: ActionRepresentationModule = (
            PearlAgent.default_action_representation_module_type()
            if action_representation_module is None
            else action_representation_module
        )
        self.policy_learner.set_action_representation_module(
            self.action_representation_module
        )

        self.policy_learner.set_history_summarization_module(
            self.history_summarization_module
        )

        # set here so replay_buffer and policy_learner are in sync
        self.replay_buffer.is_action_continuous = (
            self.policy_learner.is_action_continuous
        )
        self.replay_buffer.device = self.device

        # check that all components of the agent are compatible with each other
        pearl_agent_compatibility_check(
            self.policy_learner, self.safety_module, self.replay_buffer
        )
        self._subjective_state: SubjectiveState = None
        self._latest_action: Optional[Action] = None
        self._action_space: Optional[ActionSpace] = None

        self.policy_learner.to(self.device)
        self.history_summarization_module.to(self.device)

    def act(self, exploit: bool = False) -> Action:
        safe_action_space = self.safety_module.filter_action(self._subjective_state)

        # PolicyLearner requires all tensor inputs to be alredy on the correct device
        # before being passed to it.
        subjective_state_to_be_used = (
            torch.as_tensor(self._subjective_state).to(self.device)
            if self.policy_learner.requires_tensors  # temporary fix before abstract interfaces
            else self._subjective_state
        )

        if (
            isinstance(self._action_space, DiscreteActionSpace)
            and self.policy_learner.requires_tensors
        ):
            for a in safe_action_space.actions:
                a.to(self.device)
        self._latest_action = self.policy_learner.act(
            subjective_state_to_be_used, safe_action_space, exploit=exploit
        )
        return self._latest_action

    def observe(
        self,
        action_result: ActionResult,
    ) -> None:
        current_history = self.history_summarization_module.get_history()
        new_subjective_state = self._update_subjective_state(action_result.observation)
        new_history = self.history_summarization_module.get_history()

        # TODO: define each push with a uuid
        # TODO: currently assumes the same action space across all steps
        # need to modify ActionResults
        self.replay_buffer.push(
            current_history,
            self._latest_action,
            action_result.reward,
            new_history,
            self._action_space,  # curr_available_actions
            self._action_space,  # next_available_actions
            self._action_space,  # action_space
            action_result.done,
            action_result.cost,
        )

        self._subjective_state = new_subjective_state

    # pyre-fixme[15]: `learn` overrides method defined in `Agent` inconsistently.
    def learn(self) -> Dict[str, Any]:
        report = self.policy_learner.learn(self.replay_buffer)
        self.safety_module.learn(self.replay_buffer)

        if self.policy_learner.on_policy:
            self.replay_buffer.clear()

        return report

    def learn_batch(self, batch: TransitionBatch) -> None:
        """
        This API is often used in offline learning
        where users pass in a batch of data to train directly
        """
        batch = self.policy_learner.preprocess_batch(batch)
        self.policy_learner.learn_batch(batch)
        self.safety_module.learn_batch(batch)

    def reset(self, observation: Observation, action_space: ActionSpace) -> None:
        self.history_summarization_module.reset()
        self.history_summarization_module.to(self.device)
        self._latest_action = None
        self._subjective_state = self._update_subjective_state(observation)
        self._action_space = action_space
        self.safety_module.reset(action_space)
        self.policy_learner.reset(action_space)

    def _update_subjective_state(self, observation: Observation) -> SubjectiveState:
        if observation is None:
            return

        latest_action_representation = None
        if self._latest_action is not None:
            latest_action_representation = self.action_representation_module(
                torch.as_tensor(self._latest_action).unsqueeze(0).to(self.device)
            )
        observation_to_be_used = (
            torch.as_tensor(observation).to(self.device)
            if self.policy_learner.requires_tensors  # temporary fix before abstract interfaces
            else observation
        )

        return self.history_summarization_module.summarize_history(
            observation_to_be_used, latest_action_representation
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
