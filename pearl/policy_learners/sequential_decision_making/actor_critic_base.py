# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
from abc import abstractmethod
from typing import Any, List, Optional

import torch

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.neural_networks.common.utils import init_weights, update_target_network
from pearl.neural_networks.common.value_networks import ValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    DynamicActionActorNetwork,
    VanillaActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    make_critic,
    update_critic_target_network,
)

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.module_utils import (
    modules_have_similar_state_dict,
    optimizers_have_similar_state_dict,
)
from torch import nn, optim


class ActorCriticBase(PolicyLearner):
    """
    A base class for all actor-critic based policy learners.

    Many components that are common to all actor-critic methods have been put in this base class.
    These include:

    - actor and critic network initializations (optionally with corresponding target networks).
    - `act`, `reset` and `learn_batch` methods.
    - Utility functions used by many actor-critic methods.
    """

    def __init__(
        self,
        exploration_module: ExplorationModule,
        state_dim: int | None = None,
        actor_hidden_dims: list[int] | None = None,
        use_critic: bool = True,
        critic_hidden_dims: list[int] | None = None,
        action_space: ActionSpace | None = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        # used only for learnable history summarization module
        history_summarization_learning_rate: float = 1e-3,
        actor_network_type: type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: (
            type[ValueNetwork] | type[QValueNetwork]
        ) = VanillaQValueNetwork,
        use_actor_target: bool = False,
        use_critic_target: bool = False,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        use_twin_critic: bool = False,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        is_action_continuous: bool = False,
        on_policy: bool = False,
        action_representation_module: ActionRepresentationModule | None = None,
        actor_network_instance: ActorNetwork | None = None,
        critic_network_instance: None
        | (ValueNetwork | QValueNetwork | nn.Module) = None,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
        history_summarization_optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        super().__init__(
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
            action_space=action_space,
        )
        """
        Constructs a base actor-critic policy learner.
        """

        self._state_dim = state_dim
        self._use_actor_target = use_actor_target
        self._use_critic_target = use_critic_target
        self._use_twin_critic = use_twin_critic
        self._use_critic: bool = use_critic

        if actor_network_instance is not None:
            self._actor: nn.Module = actor_network_instance
        else:
            assert (
                state_dim is not None
            ), f"{self.__class__.__name__} requires parameter state_dim if a parameter \
            action_network_instance has not been provided."
            assert (
                actor_hidden_dims is not None
            ), f"{self.__class__.__name__} requires parameter actor_hidden_dims if a parameter \
            action_network_instance has not been provided."

            # actor network takes state as input and outputs an action vector
            self._actor: nn.Module = actor_network_type(
                input_dim=(
                    # pyre-fixme[58]: `+` is not supported for operand types `int`
                    #  and `Union[Module, Tensor]`.
                    state_dim + self.action_representation_module.representation_dim
                    if issubclass(actor_network_type, DynamicActionActorNetwork)
                    else state_dim
                ),
                hidden_dims=actor_hidden_dims,
                # pyre-fixme[6]: For 3rd argument expected `int` but got `Union[int,
                #  Module, Tensor]`.
                output_dim=(
                    1
                    if issubclass(actor_network_type, DynamicActionActorNetwork)
                    else (
                        self.action_representation_module.representation_dim
                        if self._is_action_continuous
                        else self.action_representation_module.max_number_actions
                    )
                ),
                action_space=action_space,
            )
        self._actor.apply(init_weights)
        if actor_optimizer is not None:
            self._actor_optimizer: optim.Optimizer = actor_optimizer
        else:
            # default actor optimizer
            self._actor_optimizer = optim.AdamW(
                [
                    {
                        "params": self._actor.parameters(),
                        "lr": actor_learning_rate,
                        "amsgrad": True,
                    },
                ]
            )
        self._actor_soft_update_tau = actor_soft_update_tau

        # make a copy of the actor network to be used as the actor target network
        if self._use_actor_target:
            self._actor_target: nn.Module = copy.deepcopy(self._actor)

        self._critic_soft_update_tau = critic_soft_update_tau
        if self._use_critic:
            if critic_network_instance is not None:
                self._critic: nn.Module = critic_network_instance
            else:
                assert (
                    state_dim is not None
                ), f"{self.__class__.__name__} requires parameter state_dim if a parameter \
                critic_network_instance has not been provided."
                assert (
                    critic_hidden_dims is not None
                ), f"{self.__class__.__name__} requires parameter critic_hidden_dims if a \
                parameter critic_network_instance has not been provided."

                self._critic: nn.Module = make_critic(
                    # pyre-fixme[6]: For 1st argument expected `int` but got
                    #  `Optional[int]`.
                    state_dim=self._state_dim,
                    # pyre-fixme[6]: For 2nd argument expected `Optional[int]` but
                    #  got `Union[Module, Tensor]`.
                    action_dim=self.action_representation_module.representation_dim,
                    hidden_dims=critic_hidden_dims,
                    use_twin_critic=use_twin_critic,
                    network_type=critic_network_type,
                )

            if critic_optimizer is not None:
                self._critic_optimizer: optim.Optimizer = critic_optimizer
            else:
                # default actor optimizer
                self._critic_optimizer: optim.Optimizer = optim.AdamW(
                    [
                        {
                            "params": self._critic.parameters(),
                            "lr": critic_learning_rate,
                            "amsgrad": True,
                        },
                    ]
                )
            if self._use_critic_target:
                self._critic_target: nn.Module = copy.deepcopy(self._critic)

        self._discount_factor = discount_factor
        self._history_summarization_optimizer = history_summarization_optimizer
        self._history_summarization_learning_rate = history_summarization_learning_rate
        self._actor_learning_rate: float = self._actor_optimizer.param_groups[0]["lr"]
        if self._use_critic:
            self._critic_learning_rate: float = self._critic_optimizer.param_groups[0][
                "lr"
            ]

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        """
        The history summarization module uses its own optimizer.
        """
        if self._history_summarization_optimizer is None:
            # default history summarization optimizer
            self._history_summarization_optimizer: optim.Optimizer = optim.AdamW(
                [
                    {
                        "params": value.parameters(),
                        "lr": self._history_summarization_learning_rate,
                        "amsgrad": True,
                    }
                ]
            )
        self._history_summarization_module = value

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        """
        Determines an action based on the policy network and optionally the exploration module.
        This function can operate in two modes: exploit or explore. The mode is determined by the
        `exploit` parameter.

        - If `exploit` is True, the function returns an action determined solely by the policy
        network.
        - If `exploit` is False, the function first calculates an `exploit_action` using the policy
        network. This action is then passed to the exploration module, along with additional
        arguments specific to the exploration module in use. The exploration module then generates
        an action that strikes a balance between exploration and exploitation.

        Args:
            subjective_state (SubjectiveState): Subjective state of the agent.
            available_action_space (ActionSpace): Set of eligible actions.
            exploit (bool, optional): Determines the mode of operation. If True, the function
            operates in exploit mode. If False, it operates in explore mode. Defaults to False.
        Returns:
            Action: An action (decision made by the agent in the given subjective state)
            that balances between exploration and exploitation, depending on the mode
            specified by the user. The returned action is from the available action space.
        """
        # Step 1: compute exploit_action
        # (action computed by actor network; and without any exploration)
        with torch.no_grad():
            if self._is_action_continuous:
                # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
                exploit_action = self._actor.sample_action(subjective_state)
                action_probabilities = None
            else:
                assert isinstance(available_action_space, DiscreteActionSpace)
                actions = self.action_representation_module(
                    available_action_space.actions_batch
                )
                # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
                action_probabilities = self._actor.get_policy_distribution(
                    state_batch=subjective_state,
                    available_actions=actions,
                )
                # (action_space_size)
                exploit_action_index = torch.argmax(action_probabilities)
                exploit_action = available_action_space.actions[exploit_action_index]

        # Step 2: return exploit action if no exploration,
        # else pass through the exploration module
        if exploit:
            return exploit_action

        # TODO: carefully check if safe action space is integrated with the exploration module
        return self.exploration_module.act(
            exploit_action=exploit_action,
            action_space=available_action_space,
            subjective_state=subjective_state,
            values=action_probabilities,
        )

    def reset(self, action_space: ActionSpace) -> None:
        # pyre-fixme[16]: `ActorCriticBase` has no attribute `_action_space`.
        self._action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        """
        Trains the actor and critic networks using a batch of transitions.
        This method performs the following steps:

        1. Updates the actor network with the input batch of transitions.
        2. Updates the critic network with the input batch of transitions.
        3. If using target network for critics (i.e. `use_critic_target` argument is True), the
        function updates the critic target network.
        4. If using target network for policy (i.e. `use_actor_target` argument is True), the
        function updates the actor target network.

        Note: While this method provides a general approach to actor-critic methods, specific
        algorithms may override it to introduce unique behaviors. For instance, the TD3 algorithm
        updates the actor network less frequently than the critic network.

        Args:
            batch (TransitionBatch): Batch of transitions to use for actor and critic updates.
        Returns:
            Dict[str, Any]: A dictionary containing the loss reports from the critic
            and actor updates. These can be useful to track for debugging purposes.
        """
        assert self._history_summarization_optimizer is not None
        self._history_summarization_optimizer.zero_grad()
        actor_loss = self._actor_loss(batch)
        self._actor_optimizer.zero_grad()
        """
        If the history summarization module is a neural network,
        the computation graph of this neural network is used
        to obtain both actor and critic losses.
        Without retain_graph=True, after actor_loss.backward(), the computation graph is cleared.
        After the graph is cleared, critic_loss.backward() fails.
        """
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()
        report = {"actor_loss": actor_loss.item()}
        if self._use_critic:
            self._critic_optimizer.zero_grad()
            critic_loss = self._critic_loss(batch)
            critic_loss.backward()
            self._critic_optimizer.step()
            report["critic_loss"] = critic_loss.item()
        assert self._history_summarization_optimizer is not None
        self._history_summarization_optimizer.step()

        if self._use_critic_target:
            update_critic_target_network(
                self._critic_target,
                self._critic,
                self._critic_soft_update_tau,
            )
        if self._use_actor_target:
            update_target_network(
                self._actor_target,
                self._actor,
                self._actor_soft_update_tau,
            )
        return report

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """
        Preprocesses a batch of transitions before learning on it.
        This method should be called in the learner process.
        """
        # change reward to be the lambda_constraint weighted sum of reward and cost
        if hasattr(self.safety_module, "lambda_constraint"):
            batch.reward = (
                batch.reward
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `lambda_constraint`.
                - self.safety_module.lambda_constraint * batch.cost
            )
        batch = super().preprocess_batch(batch)

        return batch

    @abstractmethod
    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the actor
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the actor network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The actor loss.
        """
        pass

    @abstractmethod
    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the critic
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the critic network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The critic loss.
        """
        pass

    def get_extra_state(self) -> dict[str, Any]:
        state_dict = {
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
        }
        if self._history_summarization_optimizer is not None:
            state_dict["history_summarization_optimizer"] = (
                self._history_summarization_optimizer.state_dict()
            )
        return state_dict

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._actor_optimizer.load_state_dict(state["actor_optimizer"])
        self._critic_optimizer.load_state_dict(state["critic_optimizer"])
        if self._history_summarization_optimizer is not None:
            self._history_summarization_optimizer.load_state_dict(
                state["history_summarization_optimizer"]
            )

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two ActorCriticBase instances for equality,
        checking attributes, networks, and exploration module.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, ActorCriticBase):
            differences.append("other is not an instance of ActorCriticBase")
        else:  # Type refinement with else block
            # Compare attributes
            if self._use_actor_target != other._use_actor_target:
                differences.append(
                    f"_use_actor_target is different: {self._use_actor_target} "
                    + f"vs {other._use_actor_target}"
                )
            if self._use_critic_target != other._use_critic_target:
                differences.append(
                    f"_use_critic_target is different: {self._use_critic_target} "
                    + f"vs {other._use_critic_target}"
                )
            if self._actor_learning_rate != other._actor_learning_rate:
                differences.append(
                    f"_actor_learning_rate is different: {self._actor_learning_rate} "
                    + f"vs {other._actor_learning_rate}"
                )
            if self._use_critic != other._use_critic:
                differences.append(
                    f"_use_critic is different: {self._use_critic} vs {other._use_critic}"
                )
            if (
                self._history_summarization_learning_rate
                != other._history_summarization_learning_rate
            ):
                differences.append(
                    "_history_summarization_learning_rate is different: "
                    + f"{self._history_summarization_learning_rate} "
                    + f"vs {other._history_summarization_learning_rate}"
                )
            if self._actor_soft_update_tau != other._actor_soft_update_tau:
                differences.append(
                    f"_actor_soft_update_tau is different: {self._actor_soft_update_tau} "
                    + f"vs {other._actor_soft_update_tau}"
                )
            if self._discount_factor != other._discount_factor:
                differences.append(
                    f"_discount_factor is different: {self._discount_factor} "
                    + f"vs {other._discount_factor}"
                )

            # Compare networks using modules_have_similar_state_dict
            if (
                reason := modules_have_similar_state_dict(self._actor, other._actor)
            ) != "":
                differences.append(f"_actor is different: {reason}")
            if self._use_critic:
                if self._critic_learning_rate != other._critic_learning_rate:
                    differences.append(
                        f"_critic_learning_rate is different: {self._critic_learning_rate} "
                        + f"vs {other._critic_learning_rate}"
                    )
                if (
                    reason := modules_have_similar_state_dict(
                        self._critic, other._critic
                    )
                ) != "":
                    differences.append(f"_critic is different: {reason}")

            # Compare target networks if they exist
            if self._use_actor_target:
                if (
                    reason := modules_have_similar_state_dict(
                        self._actor_target, other._actor_target
                    )
                ) != "":
                    differences.append(f"_actor_target is different: {reason}")
            if self._use_critic_target:
                if (
                    reason := modules_have_similar_state_dict(
                        self._critic_target, other._critic_target
                    )
                ) != "":
                    differences.append(f"_critic_target is different: {reason}")

            self.compare_optimizers(other, differences)

        return "\n".join(differences)

    def compare_optimizers(
        self, other: "ActorCriticBase", differences: List[str]
    ) -> None:
        if (
            reason := optimizers_have_similar_state_dict(
                self._actor_optimizer, other._actor_optimizer
            )
        ) != "":
            differences.append(f"_actor_optimizer is different: {reason}")
        if (
            reason := optimizers_have_similar_state_dict(
                self._critic_optimizer, other._critic_optimizer
            )
        ) != "":
            differences.append(f"_critic_optimizer is different: {reason}")
        if (self._history_summarization_optimizer is not None) != (
            other._history_summarization_optimizer is not None
        ):
            differences.append(
                "_history_summarization_optimizer is different: "
                + f"{self._history_summarization_optimizer} "
                + f"vs {other._history_summarization_optimizer}"
            )
        if (
            self._history_summarization_optimizer is not None
            and other._history_summarization_optimizer is not None
        ):
            assert (hso1 := self._history_summarization_optimizer) is not None
            assert (hso2 := other._history_summarization_optimizer) is not None
            if (
                reason := optimizers_have_similar_state_dict(
                    hso1,
                    hso2,
                )
            ) != "":
                differences.append(
                    "_history_summarization_optimizer is different: " + f"{reason}"
                )
