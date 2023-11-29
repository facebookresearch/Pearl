#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import (  # noqa E501
    SquareCBExploration,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


# The value function learning part is exactly the same as in DisjointLinUCB.


class SquareCB(DisjointLinearBandit):
    """
    SquareCB algorithm for finite action spaces.
    SquareCB requires only reward estimation, and from the empirical reward
    estimates it constructs provably efficient exploration policy.
    Unlike other approaches, SquareCB does not require uncertainty quantification.
    """

    def __init__(
        self,
        feature_dim: int,
        action_space: DiscreteActionSpace,
        gamma: int = 10,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(SquareCB, self).__init__(
            feature_dim=feature_dim,
            action_space=action_space,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=SquareCBExploration(gamma),
        )
