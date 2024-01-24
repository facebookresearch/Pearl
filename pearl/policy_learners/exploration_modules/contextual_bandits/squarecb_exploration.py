# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState

from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch.distributions.categorical import Categorical


class SquareCBExploration(ScoreExplorationBase):
    """
    SquareCB exploration model.

    Args:
        gamma (float): controls the exploration-exploitation tradeoff;
        larger leads to exploitation, smaller leads to exploration
        closer to random policy.

    Set the gamma paramer proportional to (see [1]):
        gamma ~ sqrt(T A / regret(supervised learning))
    where T is the number of time steps, A is the number of actions,
    and regret(supervised learning) is the average regret of supervised learning.

    Further information can be found in:
    [1] https://arxiv.org/abs/2002.04926
    """

    def __init__(
        self,
        gamma: float,
        reward_lb: float = 0.0,
        reward_ub: float = 1.0,
        clamp_values: bool = False,
    ) -> None:
        super(SquareCBExploration, self).__init__()
        self._gamma = gamma
        self.reward_lb = reward_lb
        self.reward_ub = reward_ub
        self.clamp_values = clamp_values

    # TODO: We should make discrete action space itself iterable
    # pyre-fixme[14]: `act` overrides method defined in `ScoreExplorationBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Optional[torch.nn.Module] = None,
        exploit_action: Optional[Action] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
    ) -> Action:
        """
        Args:
            subjective_state: vectorized or single subjective state of the agent
            for a single transition values is in shape of
            (batch_size, action_count) or (action_count)
        Returns:
            torch.Tensor: output actions index of a batch
        """

        # Calculate empirical gaps
        values = values.view(-1, action_space.n)  # (batch_size, action_space.n)
        values = self.clamp(values)
        max_val, max_indices = torch.max(values, dim=1)
        max_val.repeat(1, action_space.n)
        empirical_gaps = max_val - values

        # Construct probability distribution over actions and sample from it
        selected_actions = torch.zeros((values.size(dim=0),), dtype=torch.int)
        prob_policy = self.get_unnormalize_prob(empirical_gaps, max_val, action_space.n)
        for batch_ind in range(values.size(dim=0)):
            # Get sum of all the probabilities besides the maximum
            prob_policy[batch_ind, max_indices[batch_ind]] = 0.0
            complementary_sum = torch.sum(prob_policy)
            prob_policy[batch_ind, max_indices[batch_ind]] = 1.0 - complementary_sum
            # Sample from SquareCB update rule
            dist_policy = Categorical(prob_policy[batch_ind, :])
            selected_actions[batch_ind] = dist_policy.sample()

        return selected_actions.squeeze()

    def clamp(self, values: torch.Tensor) -> torch.Tensor:
        """
        Clamps value between min and max
        """
        if self.clamp_values:
            values = torch.clamp(values, min=self.reward_lb, max=self.reward_ub)
        return values

    def get_unnormalize_prob(
        self,
        empirical_gaps: torch.Tensor,
        max_val: float,
        action_num: Union[float, int],
    ) -> torch.Tensor:
        """
        Return unnormalized probabilities
        """
        return torch.div(1.0, action_num + self._gamma * empirical_gaps)

    # pyre-fixme[14]: `act` overrides method defined in `ScoreExplorationBase`
    #  inconsistently.
    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        exploit_action: Optional[Action] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        return values.view(-1, action_space.n)


class FastCBExploration(SquareCBExploration):
    """
    FastCB exploration model. A variation of SquareCB with a provable improved performance.
    See: https://arxiv.org/abs/2107.02237 for details.

    Assumptions: Reward is bounded. For the update rule to be valid we require bounded rewards.
    User can modify lower and upper bounds of the reward by setting reward_lb and reward_ub. clamp_values is set to True by default.

    Args:
        gamma (float): controls the exploration-exploitation tradeoff;
        larger leads to exploitation, smaller leads to exploration
        closer to random policy.

    Set the gamma paramer proportional to (see [1]):
        gamma ~ sqrt(T A / regret(supervised learning))
    where T is the number of time steps, A is the number of actions,
    and regret(supervised learning) is the average regret of supervised learning.

    """

    def __init__(
        self,
        gamma: float,
        reward_lb: float = 0.0,
        reward_ub: float = 1.0,
    ) -> None:
        super(FastCBExploration, self).__init__(
            gamma=gamma,
            reward_lb=reward_lb,
            reward_ub=reward_ub,
            clamp_values=True,
        )

    def get_unnormalize_prob(
        self,
        empirical_gaps: torch.Tensor,
        max_val: float,
        action_num: Union[float, int],
    ) -> torch.Tensor:
        """
        Return unnormalized probabilities
        """
        if max_val <= self.reward_lb:
            return torch.ones(empirical_gaps.shape) / action_num
        prob_policy = torch.div(
            (max_val - self.reward_lb),
            action_num * (max_val - self.reward_lb) + self._gamma * empirical_gaps,
        )
        return prob_policy
