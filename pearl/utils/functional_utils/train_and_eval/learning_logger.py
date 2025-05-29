# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Any, Protocol

from pearl.replay_buffers.transition import TransitionBatch


class LearningLogger(Protocol):
    """Protocol for a learning logger.
    A learning logger is a callable that takes in a dictionary of results and a step number.
    It can be used to log the results of a learning process to a database or a file.
    Args:
        results (dict[str, Any]): A dictionary of results for the batch.
        batch_index (int): has value (i - 1) after the i-th batch is processed.
        batch (TransitionBatch): The batch from which results were processed.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        results: dict[str, Any],
        batch_index: int,
        batch: TransitionBatch,
    ) -> None:
        pass


def null_learning_logger(
    results: dict[str, Any],
    batch_index: int,
    batch: TransitionBatch,
) -> None:
    """
    A learning logger that does nothing.
    """
    pass
