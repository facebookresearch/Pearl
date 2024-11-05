# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
import torch.testing as tt
from pearl.neural_networks.common.residual_wrapper import ResidualWrapper
from pearl.policy_learners.contextual_bandits.neural_bandit import LOSS_TYPES
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import (
    NeuralLinearBandit,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

NUM_EPOCHS = 1000


class TestNeuralLinearBandits(unittest.TestCase):
    def test_set_use_skip_connections(self) -> None:
        feature_dim = 16
        policy_learner = NeuralLinearBandit(
            feature_dim=feature_dim,
            hidden_dims=[16, 16],
            learning_rate=0.01,
            exploration_module=UCBExploration(alpha=0.1),
            use_skip_connections=True,
        )
        # assert ResidualWrapper is used in NeuralLinearBandit
        for child in policy_learner.model._nn_layers._model.children():
            self.assertIsInstance(child, ResidualWrapper)

    # pyre-fixme[3]: Return type must be annotated.
    def test_state_dict(self):
        # There has been discussions and debating on how to support state dict of policy learner
        # This unittest is to ensure regardless of solution, this functionality needs to be there
        # init a policy learn and learn once to get some random value
        feature_dim = 15
        batch_size = feature_dim * 4
        policy_learner = NeuralLinearBandit(
            feature_dim=feature_dim,
            hidden_dims=[32, 32],
            learning_rate=0.01,
            exploration_module=UCBExploration(alpha=0.1),
        )
        state = torch.randn(batch_size, feature_dim)
        batch = TransitionBatch(
            state=state,
            action=torch.zeros_like(state[:, -1:]),  # empty dummy values, not used
            # y = sum of state
            reward=state.sum(-1, keepdim=True),
            weight=torch.ones(batch_size, 1),
        )
        policy_learner.learn_batch(batch)

        # init another policy learner and use load_state_dict to set
        copy_policy_learner = NeuralLinearBandit(
            feature_dim=feature_dim,
            hidden_dims=[32, 32],
            learning_rate=0.01,
            exploration_module=UCBExploration(alpha=0.1),
        )
        copy_policy_learner.load_state_dict(policy_learner.state_dict())

        # assert and check if they are the same
        tt.assert_close(
            copy_policy_learner.model._linear_regression_layer._A,
            policy_learner.model._linear_regression_layer._A,
            rtol=0.0,
            atol=0.0,
        )

        tt.assert_close(
            copy_policy_learner.model._linear_regression_layer._b,
            policy_learner.model._linear_regression_layer._b,
            rtol=0.0,
            atol=0.0,
        )

        for p1, p2 in zip(
            copy_policy_learner.model._nn_layers.parameters(),
            policy_learner.model._nn_layers.parameters(),
        ):
            tt.assert_close(p1.to(p2.device), p2, rtol=0.0, atol=0.0)

    def _get_null_batch(self, feature_dim: int) -> TransitionBatch:
        return TransitionBatch(  # cf. DisjointBanditContainer._get_null_batch
            state=torch.zeros(1, feature_dim, dtype=torch.float),
            action=torch.empty(1, 0, dtype=torch.float),
            reward=torch.zeros(1, 1, dtype=torch.float),
            weight=torch.zeros(1, 1, dtype=torch.float),
        )

    # currently test support mse, mae, cross_entropy
    # separate loss_types into inddividual test cases to make it easier to debug.
    def test_neural_linucb_mse_loss(self) -> None:
        for loss_type in list(LOSS_TYPES.keys()):
            if loss_type == "mse":
                self.neural_linucb(
                    loss_type=loss_type,
                    epochs=NUM_EPOCHS,
                    output_activation_name="linear",
                )

    def test_neural_linucb_mae_loss(self) -> None:
        for loss_type in list(LOSS_TYPES.keys()):
            if loss_type == "mae":
                self.neural_linucb(
                    loss_type=loss_type,
                    epochs=NUM_EPOCHS,
                    output_activation_name="linear",
                )

    def test_neural_linucb_cross_entropy_loss(self) -> None:
        for loss_type in list(LOSS_TYPES.keys()):
            if loss_type == "cross_entropy":
                self.neural_linucb(
                    loss_type=loss_type,
                    epochs=NUM_EPOCHS,
                    output_activation_name="sigmoid",
                )

    def neural_linucb(
        self, loss_type: str, epochs: int, output_activation_name: str
    ) -> None:
        feature_dim = 15  # It is important to keep this different from hidden_dims
        batch_size = feature_dim * 4  # It is important to have enough data for training
        policy_learner = NeuralLinearBandit(
            feature_dim=feature_dim,
            hidden_dims=[32, 32],
            learning_rate=0.01,
            exploration_module=UCBExploration(alpha=0.1),
            dropout_ratio=0.0001,
            loss_type=loss_type,
            output_activation_name=output_activation_name,
        )
        self.assertEqual(feature_dim, policy_learner.feature_dim)
        state = torch.randn(batch_size, feature_dim)
        reward = state.sum(-1, keepdim=True)
        # linear relation between label(reward) and feature (state,action pair)
        if output_activation_name == "sigmoid":
            reward = torch.nn.Sigmoid()(reward)
            assert torch.all(reward >= 0) and torch.all(reward <= 1)

        batch = TransitionBatch(
            state=state,
            action=torch.zeros_like(state[:, -1:]),  # empty dummy values, not used
            reward=reward,
            weight=torch.ones(batch_size, 1),
        )
        losses = []
        for i in range(epochs):
            if i == 1:
                # simulate empty batch on early iter: can happen from DisjointBandit.
                losses.append(
                    policy_learner.learn_batch(self._get_null_batch(feature_dim))[
                        "loss"
                    ]
                )
                continue
            losses.append(policy_learner.learn_batch(batch)["loss"])

        if epochs >= NUM_EPOCHS:
            self.assertTrue(all(not torch.isnan(x) for x in losses))
            if loss_type == "mse":
                self.assertGreater(1e-1, losses[-1])
            elif loss_type == "mae":
                self.assertGreater(1e-1, losses[-1] ** 2)  # turn mae into mse
            elif loss_type == "cross_entropy":
                # cross_entropy (BCE) does not guarantee train loss to 0+ when labels are not 0/1
                self.assertTrue(
                    losses[-1] < losses[0], "training loss should be decreasing"
                )

        scores = policy_learner.get_scores(
            subjective_state=batch.state,
            action_space=DiscreteActionSpace(actions=list(batch.action)),
        )
        # shape should be batch_size, action_count
        self.assertEqual(scores.shape, (batch.state.shape[0], batch.action.shape[0]))

        # TEST ACT API
        action_space = DiscreteActionSpace(actions=list(batch.action))
        # act on one state
        action = policy_learner.act(
            subjective_state=state[0], available_action_space=action_space
        )
        self.assertTrue(action in range(batch_size))  # return action index
        # act on a batch of states
        action = policy_learner.act(
            subjective_state=state, available_action_space=action_space
        )
        self.assertEqual(action.shape, (batch_size,))

    def test_discounting(self) -> None:
        """
        Test discounting
        """
        feature_dim = 10
        batch_size = 100
        policy_learner = NeuralLinearBandit(
            feature_dim=feature_dim,
            hidden_dims=[16, 16],
            learning_rate=0.01,
            exploration_module=UCBExploration(alpha=0.1),
            use_skip_connections=True,
            gamma=0.95,
            apply_discounting_interval=100.0,
        )
        state = torch.randn(batch_size, feature_dim)
        batch = TransitionBatch(
            state=state,
            action=torch.zeros_like(state[:, -1:]),  # empty dummy values, not used
            # y = sum of state + sum of action
            reward=torch.exp(state.sum(-1, keepdim=True)),
            weight=torch.ones(batch_size, 1),
        )

        num_reps = 100
        for _ in range(num_reps):
            policy_learner.learn_batch(batch)

        self.assertLess(
            policy_learner.model._linear_regression_layer.A[0, 0].item(),
            # pyre-fixme[58]: `*` is not supported for operand types `int` and
            #  `Union[bool, float, int]`.
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            num_reps * torch.sum(batch.weight).item(),
        )
        self.assertLess(
            policy_learner.model._linear_regression_layer._b[0].item(),
            # pyre-fixme[58]: `*` is not supported for operand types `int` and
            #  `Union[bool, float, int]`.
            num_reps * torch.sum(batch.reward * batch.weight).item(),
        )
