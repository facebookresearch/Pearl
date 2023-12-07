# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.distributed as dist
from pearl.utils.functional_utils.python_utils import value_of_first_item


class DeviceNotFoundInModuleError(ValueError):
    pass


def get_device(module: torch.nn.Module) -> torch.device:
    """
    Get the device that a module is on.
    This is achieved by looking for non-empty parameters in the module and returning the
    device of the first parameter found.
    If no parameters are found, then we look for sub-modules and recurse down the tree
    until we find a parameter or reach the end.
    If we have neither parameters not sub-modules,
    then a DeviceNotFoundInModuleError is raised.
    """
    if (
        hasattr(module, "_parameters")
        and (first_parameter := value_of_first_item(module._parameters)) is not None
    ):
        return first_parameter.device
    elif (first_sub_module := value_of_first_item(module._modules)) is not None:
        try:
            return get_device(first_sub_module)
        except DeviceNotFoundInModuleError:
            raise DeviceNotFoundInModuleError(
                f"Could not find a device for module {module} because it "
                "has no parameters and could not find device in its first sub-module"
            )
    else:
        raise DeviceNotFoundInModuleError(
            f"Cannot determine the device for module {module}"
            "because it has neither parameters nor sub-modules"
        )


def get_pearl_device(device_id: int = -1) -> torch.device:
    if device_id != -1:
        return torch.device("cuda:" + str(device_id))

    try:
        # This is to pytorch distributed run, and should not affect
        # original implementation of this file
        local_rank = dist.get_rank()
    except Exception:
        local_rank = 0

    return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


def is_distribution_enabled() -> bool:
    return dist.is_initialized() and dist.is_available()


def get_default_device() -> torch.device:
    """
    Returns the torch default device, that is,
    the device on which factory methods without a `device`
    specification place their tensors.
    """
    return torch.tensor(0).device
