# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Any, Dict, Protocol


class LearningLogger(Protocol):
    """Protocol for a learning logger.
    A learning logger is a callable that takes in a dictionary of results and a step number.
    It can be used to log the results of a learning process to a database or a file.
    Args:
        results: A dictionary of results.
        step: The current step of the learning process.
        prefix: A prefix to add to the logged results.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, results: Dict[str, Any], step: int, prefix: str = "") -> None:
        pass


def null_learning_logger(results: Dict[str, str], step: int, prefix: str = "") -> None:
    """
    A null learning logger that does nothing.
    """
    pass
