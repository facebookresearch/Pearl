#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import __manifest__

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.contextual_bandits.policy_learners.deep_bandit import DeepBandit


def train(rank, world_size):
    feature_dim = 3
    batch_size = 3
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method=f"tcp://localhost:29502?world_size={world_size}&rank={rank}",
    )

    policy_learner = DeepBandit(
        feature_dim=feature_dim, hidden_dims=[16, 16], exploration_module=None
    )

    random_batch = TransitionBatch(
        state=torch.randn(batch_size, 1),
        action=torch.randn(batch_size, feature_dim - 1),
        reward=torch.randn(batch_size),
        weight=torch.randn(batch_size),
    )
    policy_learner.learn_batch(random_batch)

    dist.barrier()
    torch.save(policy_learner.get_extra_state(), f"/tmp/final_model_{rank}.pth")


class TestDeepBandit(unittest.TestCase):
    @unittest.skipIf(
        __manifest__.fbmake.get("build_mode", "") != "opt",
        "This test only works with opt mode",
    )
    def test_same_weights_among_different_ranks(self) -> None:
        world_size = (
            torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
        )

        mp.spawn(train, args=(world_size,), nprocs=world_size)
        weights = {}
        for i in range(world_size):
            weights[i] = torch.load(f"/tmp/final_model_{i}.pth")

        # check weights on all ranks are the same
        for i in range(1, world_size):
            for k, v in weights[0]["deep_represent_layers"].items():
                self.assertTrue(
                    torch.equal(v, weights[i]["deep_represent_layers"][k].to(v.device))
                )
