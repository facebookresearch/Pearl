# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from collections.abc import Iterable
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def value_of_first_item(d: dict[K, V]) -> V | None:
    """Returns the first item in a dictionary or None if inexistent."""
    return next(iter(d.values())) if d else None


def first_item(i: Iterable[V]) -> V | None:
    """Returns the first item of an Iterable or None if there is none."""
    try:
        return next(iter(i))
    except StopIteration:
        return None


# In the following, we create a generic function that takes a type arg_type,
# and returns an optional object of arg_type.
# Ideally, the return type would be Optional[arg_type] but this is not valid
# because at static time we do not know whether arg_type will contain a type.
# We get around this by using a type variable ArgType,
# which, being known at static time to be a type, can be used to define the return type.
# Taking this equivalency into account,
# the type of arg_type is the type of ArgType, so we define it as
# being of Type[ArgType].
# The type checker is then able to, at static time, to determine
# ArgType for any individual call (by examining the type of the actual argument
# being passed), and therefore determining the return type at static time.

ArgType = TypeVar("ArgType")


def find_argument(
    kwarg_key: str,
    arg_type: type[ArgType],
    # pyre-fixme[2]: Parameter must be annotated.
    *args,
    **kwargs,  # pyre-ignore
) -> ArgType | None:
    """
    Finds the first argument in args and kwargs that either has type `arg_type` or
    is a named argument with the given kwarg_key.
    Returns None if no such argument exists.
    """
    for arg in args:
        if isinstance(arg, arg_type):
            return arg
    for k, v in kwargs.items():
        if k == kwarg_key:
            return v
    return None
