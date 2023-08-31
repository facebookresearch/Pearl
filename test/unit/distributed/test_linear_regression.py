#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pearl.utils.linear_regression import LinearRegression


def train(rank, world_size):
    feature_dim = 3
    batch_size = 3

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:29501?world_size={world_size}&rank={rank}",
    )

    linear_regression = LinearRegression(feature_dim=feature_dim)

    feature = torch.ones(batch_size, feature_dim, device=rank)
    reward = feature.sum(-1).to(rank)
    weight = torch.ones(batch_size).to(rank)
    linear_regression.learn_batch(x=feature, y=reward, weight=weight)

    dist.barrier()
    if rank == 0:
        torch.save(linear_regression.state_dict(), "/tmp/final_model.pth")


class TestLinearRegression(unittest.TestCase):
    @unittest.skipIf(
        bool(not torch.cuda.is_available()),
        "test_reduce_all needs GPU",
    )
    def test_reduce_all(self) -> None:
        world_size = torch.cuda.device_count()
        feature_dim = 3
        batch_size = 3

        # train in multi process
        mp.spawn(train, args=(world_size,), nprocs=world_size)
        mp_state_dict = torch.load("/tmp/final_model.pth")

        # train in single process
        linear_regression = LinearRegression(feature_dim=feature_dim)
        feature = torch.ones(batch_size, feature_dim)
        reward = feature.sum(-1)
        weight = torch.ones(batch_size)
        for _ in range(world_size):
            linear_regression.learn_batch(x=feature, y=reward, weight=weight)

        sp_state_dict = linear_regression.state_dict()
        for k, v in mp_state_dict.items():
            self.assertTrue(torch.equal(v, sp_state_dict[k]))
