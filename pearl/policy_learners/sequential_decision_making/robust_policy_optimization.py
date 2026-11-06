# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RPO: uniform Gaussian-mean perturbations during PPO policy updates only.

Rahman and Xue, Robust Policy Optimization in Deep Reinforcement Learning,
https://arxiv.org/abs/2212.07536. Alpha is task-sensitive; benchmark it explicitly.
"""

import math
from pearl.policy_learners.sequential_decision_making.continuous_ppo import (
    ContinuousProximalPolicyOptimization,
)


class RobustPolicyOptimization(ContinuousProximalPolicyOptimization):
    """Perturb policy-update means while collecting with the original policy.

    ``rpo_alpha`` sets the uniform perturbation half-width. Zero recovers
    the PPO objective. Old likelihoods and deterministic evaluation remain
    unperturbed; optional target-KL stopping measures the perturbed policy.
    Choose alpha before evaluation on held-out seeds.
    """

    def __init__(self, *args, rpo_alpha: float = 0.5, **kwargs) -> None:
        if not math.isfinite(rpo_alpha) or rpo_alpha < 0:
            raise ValueError("rpo_alpha must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self._rpo_alpha = rpo_alpha
        self._checkpoint_config["rpo_alpha"] = rpo_alpha
