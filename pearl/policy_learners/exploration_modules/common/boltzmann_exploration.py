#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List
from warnings import warn

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules import ExplorationModule
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class BoltzmannExploration(ScoreExplorationBase):
    """Softmax (Boltzmann) exploration: `pi(a|s) = softmax(Q(s, a) / T)`.

    `get_scores` returns the probability distribution over actions, not raw
    Q-values. `act` samples an action stochastically from that distribution
    via `torch.multinomial` -- this is the actual Boltzmann exploration
    behavior, in contrast to the base `ScoreExplorationBase.act` which would
    argmax the scores (greedy w.r.t. softmax = greedy w.r.t. Q).

    Use `temperature > 0`. Lower T => sharper (more greedy); higher T => flatter
    (more exploration). T=1.0 is a reasonable default; tune from validation.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"BoltzmannExploration requires temperature > 0, got {temperature}"
            )
        self.temperature: float = temperature

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        if exploit_action is not None:
            raise ValueError("exploit_action shouldn't be used. use `values` instead")
        assert isinstance(action_space, DiscreteSpace)
        assert values is not None
        return torch.softmax(
            values.reshape(-1, action_space.n) / self.temperature, dim=-1
        )

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        action_availability_mask: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        """Sample an action stochastically from `softmax(Q / T)`.

        Overrides the base `ScoreExplorationBase.act` (which argmaxes the
        score distribution). For Boltzmann exploration the intended behavior
        is to sample, so that the realized action distribution at decision
        time actually matches the published `softmax(Q/T)` propensities.
        Without this override, the class name promises softmax sampling but
        the runtime policy is greedy (since `argmax(softmax(x)) == argmax(x)`).

        Returns: action tensor of shape `(batch_size, action_dim)`.
        """
        if exploit_action is not None:
            warn(
                "exploit_action shouldn't be used. use `values` instead",
                DeprecationWarning,
            )
            return exploit_action

        assert isinstance(action_space, DiscreteActionSpace)
        assert values is not None

        scores = self.get_scores(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=representation,
        )  # shape: (batch_size, action_count)

        # Mask unavailable actions and renormalize before sampling.
        if action_availability_mask is not None:
            scores = scores * action_availability_mask
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sample one action index per batch row from the categorical distribution.
        action_index_batch = torch.multinomial(scores, num_samples=1).squeeze(-1)

        actions = torch.nn.functional.embedding(
            action_index_batch, action_space.actions_batch
        )
        return actions

    def compare(self, other: ExplorationModule) -> str:
        differences: List[str] = []
        differences.append(super().compare(other))
        if not isinstance(other, BoltzmannExploration):
            differences.append("other is not an instance of BoltzmannExploration")
        elif self.temperature != other.temperature:
            differences.append(
                f"temperature is different: {self.temperature} vs {other.temperature}"
            )
        return "\n".join(differences)
