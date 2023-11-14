from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState

from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace
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

    def __init__(self, gamma: float) -> None:
        super(SquareCBExploration, self).__init__()
        self._gamma = gamma

    # TODO: We should make discrete action space itself iterable
    # pyre-fixme[14]: `act` overrides method defined in `ScoreExplorationBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Optional[torch.nn.Module] = None,
        exploit_action: Action = None,
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
        max_val, max_indices = torch.max(values, dim=1)
        max_val.repeat(1, action_space.n)
        empirical_gaps = max_val - values

        # Construct probability distribution over actions and sample from it
        selected_actions = torch.zeros((values.size(dim=0),), dtype=torch.int)
        prob_policy = torch.div(1.0, action_space.n + self._gamma * empirical_gaps)
        for batch_ind in range(values.size(dim=0)):
            # Get sum of all the probabilities besides the maximum
            prob_policy[batch_ind, max_indices[batch_ind]] = 0.0
            complementary_sum = torch.sum(prob_policy)
            prob_policy[batch_ind, max_indices[batch_ind]] = 1.0 - complementary_sum
            # Sample from SquareCB update rule
            dist_policy = Categorical(prob_policy[batch_ind, :])
            selected_actions[batch_ind] = dist_policy.sample()

        return selected_actions.squeeze()

    # pyre-fixme[14]: `act` overrides method defined in `ScoreExplorationBase`
    #  inconsistently.
    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        exploit_action: Action = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        return values.view(-1, action_space.n)
