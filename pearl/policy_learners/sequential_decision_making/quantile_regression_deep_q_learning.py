# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

# import torch.optim as optim
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QuantileQValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.quantile_regression_deep_td_learning import (
    QuantileRegressionDeepTDLearning,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import optim


class QuantileRegressionDeepQLearning(QuantileRegressionDeepTDLearning):
    """
    Quantile Regression based Deep Q Learning Policy Learner

    Notes:
        - Support for offline learning by adding a conservative loss to the
            quantile regression based distributional
            temporal difference loss has not been added (literature does not seem to have that)
        - To do: Add support for input a network instance
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: list[int] | None = None,
        num_quantiles: int = 10,
        exploration_module: ExplorationModule | None = None,
        on_policy: bool = False,
        learning_rate: float = 5 * 0.0001,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.05,
        action_representation_module: ActionRepresentationModule | None = None,
        network_type: type[QuantileQValueNetwork] = QuantileQValueNetwork,
        network_instance: QuantileQValueNetwork | None = None,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        assert isinstance(action_space, DiscreteActionSpace)
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            on_policy=on_policy,
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else EGreedyExploration(0.10)
            ),
            hidden_dims=hidden_dims,
            num_quantiles=num_quantiles,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            soft_update_tau=soft_update_tau,
            network_type=network_type,
            network_instance=network_instance,
            action_representation_module=action_representation_module,
            optimizer=optimizer,
        )

    # QR-DQN is based on QuantileRegressionDeepTDLearning class.
    @torch.no_grad()
    def _get_next_state_quantiles(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_available_actions_batch = (
            batch.next_available_actions
        )  # shape: (batch_size x action_space_size x action_dim)

        next_unavailable_actions_mask_batch = (
            batch.next_unavailable_actions_mask
        )  # shape: (batch_size x action_space_size)

        assert next_state_batch is not None
        assert isinstance(self._action_space, DiscreteActionSpace)

        """
        Step 1: get quantiles for all possible actions in the batch
            - output shape: (batch_size x action_space_size x num_quantiles)
        """
        assert next_available_actions_batch is not None
        next_state_action_quantiles = self._Q_target.get_q_value_distribution(
            next_state_batch, next_available_actions_batch
        )

        # get q values from a q value distribution under a risk metric
        # instead of using the 'get_q_values' method of the QuantileQValueNetwork,
        # we invoke a method from the risk sensitive safety module
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `get_q_values_under_risk_metric`.
        next_state_action_values = self.safety_module.get_q_values_under_risk_metric(
            next_state_batch, next_available_actions_batch, self._Q_target
        )  # shape: (batch_size, action_space_size)

        # make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[next_unavailable_actions_mask_batch] = -float("inf")

        """
        Step 2: choose the greedy action for each state
        """
        greedy_action_idx = torch.argmax(next_state_action_values, dim=-1).unsqueeze(-1)

        """
        Step 3: get quantiles corresponding to greedy action index using torch.gather
            - as the shape of next_state_action_quantiles is
              (batch_size x action_space_size x num_quantiles),
              and the shape of greedy_action_idx is (batch_size x 1),
            - we need to expand the shape of greedy_action_idx along the last dimension for
              broadcasting
        """
        quantiles_greedy_action = torch.gather(
            input=next_state_action_quantiles,
            dim=1,
            index=greedy_action_idx.unsqueeze(-1).expand(
                -1, -1, next_state_action_quantiles.shape[-1]
            ),  # expands shape to (batch_size x 1 x num_quantiles)
        )
        return quantiles_greedy_action.view(batch_size, -1)  # shape: (batch_size, N)

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two QuantileRegressionDeepQLearning instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.extend(super().compare(other))

        if not isinstance(other, QuantileRegressionDeepQLearning):
            differences.append(
                "other is not an instance of QuantileRegressionDeepQLearning"
            )

        # No additional attributes to compare in QuantileRegressionDeepQLearning

        return "\n".join(differences)
