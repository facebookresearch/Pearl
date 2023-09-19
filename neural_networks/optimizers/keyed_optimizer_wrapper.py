#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Mapping

import torch
from torchrec.optim.keyed import KeyedOptimizer


class KeyedOptimizerWrapper(KeyedOptimizer):
    """
    KeyedOptimizer takes a dict of parameters and exposes state_dict by parameter key.
    This class is convenience wrapper to take in optim_factory callable to create KeyedOptimizer

    Usually use this wrapper, when we want to export keyed information of an optimizer.

    Args:
        models: dictionary of model name to nn.Module which this optimizer is working on
        optimizer: optimizer we would like to wrap
    """

    def __init__(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizer_cls: type(torch.optim.Optimizer),
        **kwargs: Any,
    ) -> None:
        params = {
            model_name + k: v
            for model_name, model in models.items()
            for k, v in model.named_parameters()
            if v.requires_grad
        }
        self._optimizer: torch.optim.Optimizer = optimizer_cls(
            list(params.values()), **kwargs
        )
        super().__init__(params, self._optimizer.state, self._optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad()

    def step(self, closure: Any = None) -> None:
        self._optimizer.step(closure=closure)


class NoOpOptimizer(KeyedOptimizer):
    def __init__(self):
        super().__init__({}, {}, {})

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def step(self, closure: Any = None) -> None:
        pass
