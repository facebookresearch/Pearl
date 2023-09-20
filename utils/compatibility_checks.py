# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Type

from pearl.policy_learners.policy_learner import (
    DistributionalPolicyLearner,
    PolicyLearner,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.safety_modules.risk_sensitive_safety_modules import RiskSensitiveSafetyModule
from pearl.safety_modules.safety_module import SafetyModule


def pearl_agent_compatibility_check(
    policy_learner: Type[PolicyLearner],
    safety_module: Type[SafetyModule],
    replay_buffer: Type[ReplayBuffer],
) -> None:
    """
    Check if different modules of the Pearl agent are compatible with each other.
    """
    if isinstance(policy_learner, DistributionalPolicyLearner):
        if not isinstance(safety_module, RiskSensitiveSafetyModule):
            raise TypeError(
                "A distributional policy learner requires a risk-sensitive safety module."
            )
