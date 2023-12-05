# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Optional

import torch
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

    def __init__(
        self,
        constraint_value: float,
        lambda_constraint_ub_value: float,
        lambda_constraint_init_value: float = 0.0,
        lr_lambda: float = 0.001,
        batch_size: int = 256,
    ) -> None:
        super(RewardConstrainedSafetyModule, self).__init__()
        self.constraint_value = constraint_value
        self.lr_lambda = lr_lambda
        self.lambda_constraint_ub_value = lambda_constraint_ub_value
        self.lambda_constraint = lambda_constraint_init_value
        self.batch_size = batch_size
        self._action_space: Optional[ActionSpace] = None

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        if len(replay_buffer) < self.batch_size or len(replay_buffer) == 0:
            return

        batch = replay_buffer.sample(self.batch_size)
        assert isinstance(batch, TransitionBatch)
        self.constraint_lambda_update(batch, policy_learner)

    def constraint_lambda_update(
        self, batch: TransitionBatch, policy_learner: PolicyLearner
    ) -> None:

        with torch.no_grad():
            cost_q1, cost_q2 = policy_learner.cost_critic.get_q_values(
                state_batch=batch.state,
                action_batch=policy_learner._actor.sample_action(batch.state),
            )
            cost_q = torch.maximum(cost_q1, cost_q2)
            cost_q = cost_q.mean().item()

        lambda_update = self.lambda_constraint + self.lr_lambda * (
            cost_q * (1 - policy_learner.cost_discount_factor) - self.constraint_value
        )
        lambda_update = max(lambda_update, 0.0)
        lambda_update = min(lambda_update, self.lambda_constraint_ub_value)
        self.lambda_constraint = lambda_update

    def filter_action(
        self, subjective_state: SubjectiveState, action_space: ActionSpace
    ) -> ActionSpace:
        return action_space

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space
