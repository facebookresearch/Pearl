# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import io

import torch
from later.unittest import TestCase
from pearl.test.unit.with_pytorch.test_agent import TestAgentWithPyTorch
from torch import nn


def save_and_load_state_dict(origin: nn.Module, destination: nn.Module) -> None:
    """
    Save and load origin's state dict onto destination.
    """
    buffer = io.BytesIO()
    torch.save(origin.state_dict(), buffer)
    buffer.seek(0)
    loaded_state_dict = torch.load(buffer, weights_only=False)
    destination.load_state_dict(loaded_state_dict, strict=True)


class TestSerialization(TestCase):
    def test_serialization(self) -> None:
        test_agent = TestAgentWithPyTorch()
        for (
            agent_type,
            new_agent_function,
            trained_agent_function,
        ) in test_agent.get_agent_makers():
            new_agent, _ = new_agent_function()
            trained_agent, _ = trained_agent_function()
            print(f"Testing serialization for {agent_type}")
            self.assertNotEqual(new_agent.compare(trained_agent), "")
            save_and_load_state_dict(trained_agent, new_agent)
            differences = new_agent.compare(trained_agent)
            if differences != "":
                print(f"Found differences in {agent_type}:\n{differences}\n-----\n")
            self.assertEqual(differences, "")
