# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, List

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
    SubjectiveState,
)
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
    DEFAULT_ACTION_SPACE,
)
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.action_utils import (
    concatenate_actions_to_state,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class LinearBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with Linear Policy.

    This class implements a policy learner for a contextual bandit problem where the policy is
    linear and learned through linear regression.
    See the documentation of LinearRegression for more details on the underlying model.

    It furthermore allows for _discounting_. This provides the model with the ability
    to "forget" old data and adjust to a new data distribution in a non-stationary
    environment. The discounting is applied periodically and consists of multiplying
    the underlying linear system matrices A and b (the model's weights) by gamma
    (the discounting multiplier). The discounting period is controlled by
    apply_discounting_interval, which consists of the number of inputs to be
    processed between different rounds of discounting. Note that, because inputs
    are weighted,  apply_discounting_interval is more precisely described as
    the sum of weights of inputs that need to be processed before
    discounting takes place again. This is expressed in pseudo-code as
    ```
    if apply_discounting_interval > 0 and (
            sum_weights - sum_weights_when_last_discounted
            >= apply_discounting_interval:
        A *= discount factor
        b *= discount factor
    ```
    To disable discounting, simply set gamma to 1.

    The learner also supports exploration modules for acting based on learned policies.

    Attributes:
        model (LinearRegression): Linear regression model used for learning.
        last_sum_weight_when_discounted (float): The counter for the last data point
                                                 when discounting was applied.

    Args:
        feature_dim (int): Dimension of the feature space.
        exploration_module (Optional[ExplorationModule]): module for exploring actions.
        l2_reg_lambda (float, default 1.0): L2 regularization parameter for the linear
                                            regression model.
        gamma (float, default 1.0): the discounting factor.
        apply_discounting_interval (float, default 0): number of (weighted) observations for
                                   applying discounting to the data points. Set to 0.0 to disable.
        force_pinv (float, default False): if True, we will always use pseudo-inversion to invert
                                   the A matrix. If False, we will first try to use regular
                                   matrix inversion.
                                   If it fails, we will fallback to pseudo-inverse.
        training_rounds (int): number of training rounds.
        batch_size (int, default 128): size of the batches used during training.
        action_representation_module (Optional[ActionRepresentationModule], default identity):
                                     module for representing actions.
    """

    def __init__(
        self,
        feature_dim: int,
        exploration_module: ExplorationModule | None = None,
        l2_reg_lambda: float = 1.0,
        gamma: float = 1.0,
        apply_discounting_interval: float = 0.0,
        force_pinv: bool = False,
        training_rounds: int = 100,
        batch_size: int = 128,
        action_representation_module: ActionRepresentationModule | None = None,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
        )
        self.model = LinearRegression(
            feature_dim=feature_dim,
            l2_reg_lambda=l2_reg_lambda,
            gamma=gamma,
            force_pinv=force_pinv,
        )
        self.apply_discounting_interval = apply_discounting_interval
        self.last_sum_weight_when_discounted = 0.0

    def _maybe_apply_discounting(self) -> None:
        """
        Check if it's time to apply discounting and do so if it's time.
        Discounting is applied after every N data points (weighted) are processed.

        `self.last_sum_weight_when_discounted` stores the data point counter when discounting was
            last applied.
        `self.model._sum_weight.item()` is the current data point counter
        """
        if (self.apply_discounting_interval > 0) and (
            self.model._sum_weight.item() - self.last_sum_weight_when_discounted
            >= self.apply_discounting_interval
        ):
            self.model.apply_discounting()
            self.last_sum_weight_when_discounted = self.model._sum_weight.item()

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        """
        A <- A + x*x.t
        b <- b + r*x
        """
        expected_values = batch.reward
        batch_weight = (
            batch.weight
            if batch.weight is not None
            else torch.ones_like(expected_values)
        )
        x = torch.cat([batch.state, batch.action], dim=1)
        self.model.learn_batch(
            x=x,
            y=batch.reward,
            weight=batch.weight,
        )
        self._maybe_apply_discounting()
        predicted_values = self.model(x)
        return {
            "label": expected_values,
            "prediction": predicted_values,
            "weight": batch_weight,
        }

    # pyre-fixme[14]: `act` overrides method defined in `ContextualBanditBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        action_availability_mask: torch.Tensor | None = None,
        exploit: bool = False,
    ) -> Action:
        """
        Args:
            subjective_state: state will be applied to different action vectors in action_space
            available_action_space: contains a list of action vectors.
                                    Currently, only static spaces are supported.
        Return:
            action index chosen given state and action vectors
        """
        # It doesnt make sense to call act if we are not working with action vector
        assert (
            self.exploration_module is not None
        ), "exploration module must be set to call act()"
        action_count = available_action_space.n
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=available_action_space,
            action_representation_module=self.action_representation_module,
        )
        values = self.model(new_feature)  # (batch_size, action_count)
        assert values.shape == (new_feature.shape[0], action_count)
        return self.exploration_module.act(
            subjective_state=new_feature,
            action_space=available_action_space,
            values=values,
            action_availability_mask=action_availability_mask,
            representation=self.model,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = DEFAULT_ACTION_SPACE,
    ) -> torch.Tensor:
        """
        Returns:
            UCB scores when exploration module is UCB
            Shape is (batch)
        """
        feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            action_representation_module=self.action_representation_module,
        )
        assert isinstance(self.exploration_module, ScoreExplorationBase)
        return self.exploration_module.get_scores(
            subjective_state=feature,
            values=self.model(feature),
            action_space=action_space,
            representation=self.model,
        ).squeeze(-1)

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        # currently linear bandit algorithm does not update
        # parameters of the history summarization module
        self._history_summarization_module = value

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two LinearBandit instances for equality,
        checking attributes, model, and exploration module.

        Args:
          other: The other ContextualBanditBase to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, LinearBandit):
            differences.append("other is not an instance of LinearBandit")
        else:  # Type refinement with else block
            # Compare attributes
            if self.apply_discounting_interval != other.apply_discounting_interval:
                differences.append(
                    f"apply_discounting_interval is different: {self.apply_discounting_interval} "
                    + f"vs {other.apply_discounting_interval}"
                )
            if (
                self.last_sum_weight_when_discounted
                != other.last_sum_weight_when_discounted
            ):
                differences.append(
                    "last_sum_weight_when_discounted is different: "
                    + f"{self.last_sum_weight_when_discounted} "
                    + f"vs {other.last_sum_weight_when_discounted}"
                )

            # Compare models using their compare method
            if (reason := self.model.compare(other.model)) != "":
                differences.append(f"model is different: {reason}")

        return "\n".join(differences)
