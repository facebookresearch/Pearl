# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    EnsembleQValueNetwork,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.module_utils import modules_have_similar_state_dict


class DeepExploration(ExplorationModule):
    r"""An exploration strategy that follows a policy based on a randomly
    drawn value function (from its posterior distribution), an idea that was
    developed in [1, 2, 3]. The implementation is based off of [3] and uses an
    ensemble of Q-value functions.

    [1] Ian Osband, Daniel Russo, and Benjamin Van Roy, (More) efficient reinforcement
        learning via posterior sampling. Advances in Neural Information Processing
        Systems, 2013. https://arxiv.org/abs/1306.0940.
    [2] Ian Osband, Benjamin Van Roy, Daniel Russo, and Zheng Wen, Deep exploration
        via randomized value functions. Journal of Machine Learning Research, 2019.
        https://arxiv.org/abs/1703.07608.
    [3] Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin
        Vay Roy, Deep exploration via bootstrapped DQN. Advances in Neural
        Information Processing Systems, 2016. https://arxiv.org/abs/1602.04621.

        Args:
            q_ensemble_network (EnsembleQValueNetwork): A network that outputs
                a tensor of shape (num_samples, num_actions) where each row is
                the Q-value of taking each possible action.
    """

    def __init__(
        self,
        q_ensemble_network: EnsembleQValueNetwork,
        action_representation_module: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.q_ensemble_network = q_ensemble_network
        self.action_representation_module = action_representation_module

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action | None = None,
        values: torch.Tensor | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        assert isinstance(action_space, DiscreteActionSpace)

        actions = action_space.actions_batch.to(subjective_state.device)
        # (action_space_size, action_dim)

        actions = self.action_representation_module(actions).unsqueeze(
            0
        )  # (1 x action_space_size x action_dim)

        with torch.no_grad():
            q_values = self.q_ensemble_network.get_q_values(
                state_batch=subjective_state.unsqueeze(0),
                action_batch=actions,
                z=self.q_ensemble_network._model.z,
                persistent=True,
            )
            # this does a forward pass since all available
            # actions are already stacked together

        action_index = torch.argmax(q_values)
        action = action_space.actions[action_index]
        return action

    def reset(self) -> None:  # noqa: B027
        # sample a new epistemic index (i.e., a Q-network) at the beginning of a
        # new episode for temporally consistent exploration
        self.q_ensemble_network.resample_epistemic_index()

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two DeepExploration instances for equality,
        checking attributes (q_ensemble_network and action_representation_module).

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, DeepExploration):
            differences.append("other is not an instance of DeepExploration")
        else:
            # Compare q_ensemble_network using modules_have_similar_state_dict
            if (
                reason := modules_have_similar_state_dict(
                    self.q_ensemble_network, other.q_ensemble_network
                )
            ) != "":
                differences.append(f"q_ensemble_network is different: {reason}")

            # Compare action_representation_module using modules_have_similar_state_dict
            if (
                reason := modules_have_similar_state_dict(
                    self.action_representation_module,
                    other.action_representation_module,
                )
            ) != "":
                differences.append(
                    f"action_representation_module is different: {reason}"
                )

        return "\n".join(differences)
