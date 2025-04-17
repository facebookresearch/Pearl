# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from copy import deepcopy
from typing import Any, List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.common.utils import update_target_network
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    EnsembleQValueNetwork,
)
from pearl.policy_learners.exploration_modules.sequential_decision_making.deep_exploration import (
    DeepExploration,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers.transition import (
    filter_batch_by_bootstrap_mask,
    TransitionBatch,
    TransitionWithBootstrapMaskBatch,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.module_utils import modules_have_similar_state_dict
from torch import optim, Tensor


class BootstrappedDQN(DeepQLearning):
    r"""Bootstrapped DQN, proposed by [1], is an extension of DQN that uses
    the so-called "deep exploration" mechanism. The main idea is to keep
    an ensemble of `K` Q-value networks and on each episode, one of them is
    sampled and the greedy policy associated with that network is used for
    exploration.

    [1] Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin
        Van Roy, Deep exploration via bootstrapped DQN. Advances in Neural
        Information Processing Systems, 2016. https://arxiv.org/abs/1602.04621.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        q_ensemble_network: EnsembleQValueNetwork,
        discount_factor: float = 0.99,
        learning_rate: float = 0.001,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 1.0,
        action_representation_module: ActionRepresentationModule | None = None,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        assert isinstance(action_space, DiscreteActionSpace)
        if action_representation_module is None:
            action_representation_module = IdentityActionRepresentationModule(
                max_number_actions=action_space.n,
                representation_dim=action_space.action_dim,
            )
        PolicyLearner.__init__(
            self=self,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=DeepExploration(
                q_ensemble_network, action_representation_module
            ),
            on_policy=False,
            is_action_continuous=False,
            action_representation_module=action_representation_module,
        )
        self._action_space = action_space
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._target_update_freq = target_update_freq
        self._soft_update_tau = soft_update_tau
        self._Q = q_ensemble_network
        self._Q_target: EnsembleQValueNetwork = deepcopy(self._Q)
        if optimizer is not None:
            self._optimizer: optim.Optimizer = optimizer
        else:
            self._optimizer = optim.AdamW(
                self._Q.parameters(), lr=self._learning_rate, amsgrad=True
            )

    @property
    def ensemble_size(self) -> int:
        return self._Q.ensemble_size

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        if not isinstance(batch, TransitionWithBootstrapMaskBatch):
            raise TypeError(
                f"{type(self).__name__} requires a batch of type "
                f"`TransitionWithBootstrapMaskBatch`, but got {type(batch)}."
            )
        loss_ensemble = torch.tensor(0.0).to(batch.device)
        mask = batch.bootstrap_mask
        for z in range(self.ensemble_size):
            z = torch.tensor(z).to(batch.device)
            # if this batch doesn't have any items for the z-th ensemble, move on
            if mask is None or mask[:, z].sum() == 0:
                continue

            # filter the batch to only the transitions belonging to ensemble `z`
            batch_filtered = filter_batch_by_bootstrap_mask(batch=batch, z=z)
            state_action_values = self._Q.get_q_values(
                state_batch=batch_filtered.state,
                action_batch=batch_filtered.action,
                curr_available_actions_batch=batch_filtered.curr_available_actions,
                z=z,
            )

            # compute the Bellman target
            expected_state_action_values = (
                self._get_next_state_values(
                    batch=batch_filtered, batch_size=batch_filtered.state.shape[0], z=z
                )
                * self._discount_factor
                * (1 - batch_filtered.terminated.float())
            ) + batch_filtered.reward  # (batch_size), r + gamma * V(s)

            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values)
            loss_ensemble += loss

        # Optimize the model
        self._optimizer.zero_grad()
        loss_ensemble.backward()
        self._optimizer.step()

        # Target Network Update
        if (self._training_steps + 1) % self._target_update_freq == 0:
            update_target_network(self._Q_target, self._Q, self._soft_update_tau)

        return {"loss": loss_ensemble.mean().item()}

    def reset(self, action_space: ActionSpace) -> None:
        # Reset the `DeepExploration` module, which will resample the epistemic index.
        self.exploration_module.reset()

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # Fix the available action space.
        assert isinstance(available_action_space, DiscreteActionSpace)
        with torch.no_grad():
            batched_actions_representation = self.action_representation_module(
                available_action_space.actions_batch.to(subjective_state)
            ).unsqueeze(0)  # (1 x action_space_size x action_dim)

            q_values = self._Q.get_q_values(
                subjective_state.unsqueeze(0),  # (1 x state_dim)
                batched_actions_representation,
                z=self._Q._model.z,
            )  # (1 x action_space_size)
            # this does a forward pass since all avaialble
            # actions are already stacked together

            exploit_action_index = torch.argmax(q_values)
            exploit_action = available_action_space.actions[exploit_action_index]

        if exploit:
            return exploit_action

        assert self.exploration_module is not None
        return self.exploration_module.act(
            subjective_state=subjective_state,
            action_space=available_action_space,
            exploit_action=exploit_action,
            values=q_values,
        )

    @torch.no_grad()
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int, z: Tensor
    ) -> torch.Tensor:
        assert batch.next_state is not None
        assert isinstance(self._action_space, DiscreteActionSpace)
        assert batch.next_available_actions is not None
        assert batch.next_unavailable_actions_mask is not None

        # for dueling, this does a forward pass; since the batch of next available
        # actions is already input
        next_state_action_values = self._Q.get_q_values(
            state_batch=batch.next_state,  # (batch_size x state_dim)
            # (batch_size x action_space_size x action_dim)
            action_batch=batch.next_available_actions,
            z=z,
        )  # (batch_size x action_space_size)

        target_next_state_action_values = self._Q_target.get_q_values(
            # pyre-fixme[6]: In call `EnsembleQValueNetwork.get_q_values`,
            # for argument `state_batch`, expected `Tensor` but got `Optional[Tensor]`
            state_batch=batch.next_state,
            # pyre-fixme[6]: In call `EnsembleQValueNetwork.get_q_values`,
            # for argument `action_batch`, expected `Tensor` but got `Optional[Tensor]`
            action_batch=batch.next_available_actions,
            z=z,
        )

        # Make sure that unavailable actions' Q values are assigned to -inf
        next_state_action_values[batch.next_unavailable_actions_mask] = -float("inf")

        # Get argmax actions indices
        argmax_actions = next_state_action_values.max(1)[1]  # (batch_size)
        return target_next_state_action_values[
            torch.arange(batch_size), argmax_actions
        ]  # (batch_size)

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two BootstrappedDQN instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        # Do not invoke super().compare() because this class is anomalous in
        # its __init__; it does not invoke super().__init__.
        # differences.append(super().compare(other))

        if not isinstance(other, BootstrappedDQN):
            differences.append("other is not an instance of BootstrappedDQN")
        else:  # Type refinement with else block
            # Compare attributes from PolicyLearner
            # (since BootstrappedDQN doesn't call super().__init__)
            if self._training_rounds != other._training_rounds:
                differences.append(
                    f"_training_rounds is different: {self._training_rounds} "
                    + f"vs {other._training_rounds}"
                )
            if self._batch_size != other._batch_size:
                differences.append(
                    f"_batch_size is different: {self._batch_size} vs "
                    + f"{other._batch_size}"
                )
            if self.on_policy != other.on_policy:
                differences.append(
                    f"on_policy is different: {self.on_policy} vs "
                    + f"{other.on_policy}"
                )
            if self._is_action_continuous != other._is_action_continuous:
                differences.append(
                    f"_is_action_continuous is different: {self._is_action_continuous} vs "
                    + f"{other._is_action_continuous}"
                )

            # Compare attributes from BootstrappedDQN
            if self._learning_rate != other._learning_rate:
                differences.append(
                    f"_learning_rate is different: {self._learning_rate} vs {other._learning_rate}"
                )
            if self._discount_factor != other._discount_factor:
                differences.append(
                    f"_discount_factor is different: {self._discount_factor} vs "
                    + f"{other._discount_factor}"
                )
            if self._target_update_freq != other._target_update_freq:
                differences.append(
                    f"_target_update_freq is different: {self._target_update_freq} vs "
                    + f"{other._target_update_freq}"
                )
            if self._soft_update_tau != other._soft_update_tau:
                differences.append(
                    f"_soft_update_tau is different: {self._soft_update_tau} vs "
                    + f"{other._soft_update_tau}"
                )

            # Compare Q-networks and target Q-networks using modules_have_similar_state_dict
            if (reason := modules_have_similar_state_dict(self._Q, other._Q)) != "":
                differences.append(f"_Q is different: {reason}")
            if (
                reason := modules_have_similar_state_dict(
                    self._Q_target, other._Q_target
                )
            ) != "":
                differences.append(f"_Q_target is different: {reason}")

            # Compare exploration modules
            if self.exploration_module is None:
                if other.exploration_module is not None:
                    differences.append(
                        "exploration_module is different: None vs not None"
                    )
            elif (
                reason := self.exploration_module.compare(other.exploration_module)
            ) != "":
                differences.append(f"exploration_module is different: {reason}")

        return "\n".join(differences)
