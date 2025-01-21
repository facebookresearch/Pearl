# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.deep_td_learning import (
    DeepTDLearning,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import optim


class DeepQLearning(DeepTDLearning):
    """
    Deep Q Learning Policy Learner
    """

    def __init__(
        self,
        action_space: ActionSpace | None = None,
        hidden_dims: list[int] | None = None,
        exploration_module: ExplorationModule | None = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        training_rounds: int = 10,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.75,  # a value of 1 indicates no soft updates
        is_conservative: bool = False,
        conservative_alpha: float | None = 2.0,
        state_dim: int | None = None,
        network_type: type[QValueNetwork] = VanillaQValueNetwork,
        action_representation_module: ActionRepresentationModule | None = None,
        network_instance: QValueNetwork | None = None,
        optimizer: Optional[optim.Optimizer] = None,
        **kwargs: Any,
    ) -> None:
        """Constructs a DeepQLearning policy learner. DeepQLearning is based on DeepTDLearning
        class and uses `act` and `learn_batch` methods of that class. We only implement the
        `get_next_state_values` function to compute the bellman targets using Q-learning.

        Args:
            state_dim: Dimension of the observation space.
            action_space (ActionSpace, optional): Action space of the problem. It is kept optional
                to allow for the use of dynamic action spaces (both `learn_batch` and `act`
                functions). Defaults to None.
            hidden_dims (List[int], optional): Hidden dimensions of the default `QValueNetwork`
                (taken to be `VanillaQValueNetwork`). Defaults to None.
            exploration_module (ExplorationModule, optional): Optional exploration module to
                trade-off between exploitation and exploration. Defaults to None.
            learning_rate (float): Learning rate for AdamW optimizer. Defaults to 0.001.
                Note: We use AdamW by default for all value based methods.
            discount_factor (float): Discount factor for TD updates. Defaults to 0.99.
            training_rounds (int): Number of gradient updates per environment step.
                Defaults to 10.
            batch_size (int): Sample size for mini-batch gradient updates. Defaults to 128.
            target_update_freq (int): Frequency at which the target network is updated.
                Defaults to 10.
            soft_update_tau (float): Coefficient for soft updates to the target networks.
                Defaults to 0.01.
            is_conservative (bool): Whether to use conservative updates for offline learning
                with conservative Q-learning (CQL). Defaults to False.
            conservative_alpha (float, optional): Alpha parameter for CQL. Defaults to 2.0.
            network_type (Type[QValueNetwork]): Network type for the Q-value network. Defaults to
                `VanillaQValueNetwork`. This means that by default, an instance of the class
                `VanillaQValueNetwork` (or the specified `network_type` class) is created and used
                for learning.
            action_representation_module (ActionRepresentationModule, optional): Optional module to
                represent actions as a feature vector. Typically specified at the agent level.
                Defaults to None.
            network_instance (QValueNetwork, optional): A network instance to be used as the
                Q-value network. Defaults to None.
                Note: This is an alternative to specifying a `network_type`. If provided, the
                specified `network_type` is ignored and the input `network_instance` is used for
                learning. Allows for custom implementations of Q-value networks.
            **kwargs: Additional arguments to be passed when using `TwoTowerNetwork`
                class as the QValueNetwork. This includes {state_output_dim (int),
                action_output_dim (int), state_hidden_dims (List[int]),
                action_hidden_dims (List[int])}, all of which are used to instantiate a
                `TwoTowerNetwork` object.

        """
        super().__init__(
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else EGreedyExploration(0.05)
            ),
            on_policy=False,
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            soft_update_tau=soft_update_tau,
            is_conservative=is_conservative,
            conservative_alpha=conservative_alpha,
            network_type=network_type,
            action_representation_module=action_representation_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            network_instance=network_instance,
            optimizer=optimizer,
            **kwargs,
        )

    @torch.no_grad()
    def get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        """
        Computes the maximum Q-value over all available actions in the next state using the target
        network. Note: Q-learning is designed to work with discrete action spaces.

        Args:
            batch (TransitionBatch): Batch of transitions. For Q learning, any transtion must have
                the 'next_state', 'next_available_actions' and the 'next_unavailable_actions_mask'
                fields set. The 'next_available_actions' and 'next_unavailable_actions_mask' fields
                implement dynamic actions spaces in Pearl.
            batch_size (int): Size of the batch.

        Returns:
            torch.Tensor: Maximum Q-value over all available actions in the next state.
        """

        assert batch.next_state is not None
        assert batch.next_available_actions is not None
        assert batch.next_unavailable_actions_mask is not None
        assert isinstance(self._action_space, DiscreteActionSpace)

        # Get Q values for each (state, action), where action \in {available_actions}
        next_state_action_values = self._Q_target.get_q_values(
            batch.next_state,  # (batch_size x state_dim)
            batch.next_available_actions,  # (batch_size x action_space_size x action_dim)
        )  # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[batch.next_unavailable_actions_mask] = -float("inf")

        # Torch.max(1) returns value, indices
        return next_state_action_values.max(1)[0]  # (batch_size)

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two DeepQLearning instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        # First, perform the comparisons from the base class
        differences.append(super().compare(other))

        if not isinstance(other, DeepQLearning):
            differences.append("other is not an instance of DeepQLearning")

        # No additional attributes to compare in DeepQLearning

        return "\n".join(differences)
