# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from Pearl.pearl.policy_learners.policy_learner import (
    DistributionalPolicyLearner,
    PolicyLearner,
)
from Pearl.pearl.policy_learners.sequential_decision_making.td3 import RCTD3
from Pearl.pearl.replay_buffers.replay_buffer import ReplayBuffer
from Pearl.pearl.safety_modules.reward_constrained_safety_module import (
    RewardConstrainedSafetyModule,
)
from Pearl.pearl.safety_modules.risk_sensitive_safety_modules import RiskSensitiveSafetyModule
from Pearl.pearl.safety_modules.safety_module import SafetyModule


def pearl_agent_compatibility_check(
    policy_learner: PolicyLearner,
    safety_module: SafetyModule,
    replay_buffer: ReplayBuffer,
) -> None:
    """
    Check if different modules of the Pearl agent are compatible with each other.
    """
    if isinstance(policy_learner, DistributionalPolicyLearner):
        if not isinstance(safety_module, RiskSensitiveSafetyModule):
            raise TypeError(
                "A distributional policy learner requires a risk-sensitive safety module."
            )

    if isinstance(safety_module, RewardConstrainedSafetyModule):
        if not isinstance(policy_learner, RCTD3):
            raise TypeError(
                "An Reward Constrained Policy Optimization safety module requires RCTD3 policy learner."
            )
