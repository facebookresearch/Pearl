# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def assert_is_tensor_like(something: object) -> torch.Tensor:
    """
    Asserts that an object is either an instance of torch.Tensor
    or torch.fx.proxy.Proxy, and returns the same object
    typed as torch.Tensor.

    This is a replacement of `isinstance(something, torch.Tensor)`
    that is more flexible and accepts torch.fx.proxy.Proxy,
    which behaves like torch.Tensor but is not a subtype of it.

    This is used to satisfy APS, an internal Meta framework
    that uses proxies for performing tracing.
    This will be eventually made unnecessary when RL types
    are better encapsulated.
    """
    assert isinstance(something, torch.Tensor) or isinstance(
        something, torch.fx.proxy.Proxy
    )
    return something  # pyre-ignore
