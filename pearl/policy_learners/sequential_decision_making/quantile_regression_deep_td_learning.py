# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.neural_networks.common.utils import update_target_network
from pearl.neural_networks.common.value_networks import QuantileQValueNetwork
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import DistributionalPolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.risk_sensitive_safety_modules import (  # noqa
    RiskNeutralSafetyModule,  # noqa
)
from pearl.utils.functional_utils.learning.loss_fn_utils import (
    compute_elementwise_huber_loss,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import optim


# TODO: Only support discrete action space problems for now and assumes Gym action space.
class QuantileRegressionDeepTDLearning(DistributionalPolicyLearner):
    """
    An Abstract Class for Quantile Regression based Deep Temporal Difference learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        on_policy: bool,
        exploration_module: ExplorationModule,
        hidden_dims: Optional[List[int]] = None,
        num_quantiles: int = 10,
        learning_rate: float = 5 * 0.0001,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.05,  # typical value for soft update
        network_type: Type[
            QuantileQValueNetwork
        ] = QuantileQValueNetwork,  # C51 might use a different network type; add that later
        network_instance: Optional[QuantileQValueNetwork] = None,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        assert isinstance(action_space, DiscreteActionSpace)
        super(QuantileRegressionDeepTDLearning, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            on_policy=on_policy,
            is_action_continuous=False,
            action_representation_module=action_representation_module,
        )

        if hidden_dims is None:
            hidden_dims = []

        self._action_space = action_space
        self._discount_factor = discount_factor
        self._target_update_freq = target_update_freq
        self._soft_update_tau = soft_update_tau
        self._num_quantiles = num_quantiles

        def make_specified_network() -> QuantileQValueNetwork:
            assert hidden_dims is not None
            return network_type(
                state_dim=state_dim,
                action_dim=action_space.n,  # pyre-ignore[16]
                hidden_dims=hidden_dims,
                num_quantiles=num_quantiles,
            )

        if network_instance is not None:
            self._Q: QuantileQValueNetwork = network_instance
            assert network_instance.state_dim == state_dim, (
                "input state dimension doesn't match network "
                "state dimension for QuantileQValueNetwork"
            )
            assert network_instance.action_dim == action_space.n, (
                "input action dimension doesn't match network "
                "action dimension for QuantileQValueNetwork"
            )
        else:
            assert hidden_dims is not None
            self._Q: QuantileQValueNetwork = make_specified_network()

        self._Q_target: QuantileQValueNetwork = copy.deepcopy(self._Q)
        self._optimizer = optim.AdamW(
            self._Q.parameters(), lr=learning_rate, amsgrad=True
        )

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        self._optimizer.add_param_group({"params": value.parameters()})
        self._history_summarization_module = value

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        assert isinstance(available_action_space, DiscreteActionSpace)
        # Fix the available action space.
        with torch.no_grad():
            states_repeated = torch.repeat_interleave(
                subjective_state.unsqueeze(0),
                available_action_space.n,
                dim=0,
            )  # (action_space_size x state_dim)

            actions = F.one_hot(torch.arange(0, available_action_space.n)).to(
                subjective_state.device
            )
            # (action_space_size, action_dim)

            # instead of using the 'get_q_values' method of the QuantileQValueNetwork,
            # we invoke a method from the risk sensitive safety module
            q_values = self.safety_module.get_q_values_under_risk_metric(
                states_repeated, actions, self._Q
            )
            exploit_action = torch.argmax(q_values).view((-1))

        if exploit:
            return exploit_action

        return self._exploration_module.act(
            subjective_state,
            available_action_space,
            exploit_action,
            q_values,
        )

    # QR DQN, QR SAC and QR SARSA will implement this differently
    @abstractmethod
    def _get_next_state_quantiles(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        pass

    # learn quantiles of q value distribution using distribution temporal
    # difference learning (specifically, quantile regression)
    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Assume N is the number of quantiles.

        - This is the learning update for the quantile q value network which,
        for each (state, action) pair, computes the quantile locations
        (theta_1(s,a), .. , theta_N(s,a)). The quantiles are fixed to be 1/N.
        - The return distribution is represented as: Z(s, a) = (1/N) * sum_{i=1}^N theta_i (s,a),
        where (theta_1(s,a), .. , theta_N(s,a)),
        which represent the quantile locations, are outouts of the QuantileQValueNetwork.
        - Loss function:
                sum_{i=1}^N E_{j} [ rho_{tau^*_i}( T theta_j(s',a*) - theta_i(s,a) ) ] - Eq (1)

            - tau^*_i is the i-th quantile midpoint ((tau_i + tau_{i-1})/2),
            - T is the distributional Bellman operator,
            - rho_tau(.) is the asymmetric quantile huber loss function,
            - theta_i and theta_j are outputs of the QuantileQValueNetwork,
              representing locations of quantiles,
            - a* is the greedy action with respect to Q values (computed from the q value
              distribution under some risk metric)

        See the parameterization in QR DQN paper: https://arxiv.org/pdf/1710.10044.pdf for details.
        """

        batch_size = batch.state.shape[0]

        """
        Step 1: a forward pass through the quantile network which gives quantile locations,
        theta(s,a), for each (state, action) pair
        """
        # a forward pass through the quantile network which gives quantile locations
        # for each (state, action) pair
        quantile_state_action_values = self._Q.get_q_value_distribution(
            state_batch=batch.state, action_batch=batch.action
        )  # shape: (batch_size, num_quantiles)

        """
        Step 2: compute Bellman target for each quantile location
            - add a dimension to the reward and (1-done) vectors so they
              can be broadcasted with the next state quantiles
        """

        with torch.no_grad():
            quantile_next_state_greedy_action_values = self._get_next_state_quantiles(
                batch, batch_size
            ) * self._discount_factor * (1 - batch.done.float()).unsqueeze(
                -1
            ) + batch.reward.unsqueeze(
                -1
            )

        """
        Step 3: pairwise distributional quantile loss:
        T theta_j(s',a*) - theta_i(s,a) for i,j in (1, .. , N)
            - output shape: (batch_size, N, N)
        """
        pairwise_quantile_loss = quantile_next_state_greedy_action_values.unsqueeze(
            2
        ) - quantile_state_action_values.unsqueeze(1)

        # elementwise huber loss smoothes the quantile loss, since it is non-smooth at 0
        huber_loss = compute_elementwise_huber_loss(pairwise_quantile_loss)

        with torch.no_grad():
            asymmetric_weight = torch.abs(
                self._Q.quantile_midpoints - (pairwise_quantile_loss < 0).float()
            )

        """
        # Step 4: compute asymmetric huber loss (also known as the quantile huber loss)
            - output shape: (batch_size, N, N)
        """
        quantile_huber_loss = asymmetric_weight * huber_loss

        """
        Step 5: compute loss to optimize: given pairwise quantile huber loss,
            - sum(dim=1) approximates the (sum_{i=1}^N [ .. ]) term in Equation (1),
            - mean() takes average over the other quantile dimension (E_j [ .. ]) and over batch
        """
        quantile_bellman_loss = quantile_huber_loss.sum(dim=1).mean()

        # optimize model (parameters of quantile q network)
        self._optimizer.zero_grad()
        quantile_bellman_loss.backward()
        self._optimizer.step()

        # target network update
        if (self._training_steps + 1) % self._target_update_freq == 0:
            update_target_network(self._Q_target, self._Q, self._soft_update_tau)

        return {
            "loss": torch.abs(
                quantile_state_action_values - quantile_next_state_greedy_action_values
            )
            .mean()
            .item()
        }
