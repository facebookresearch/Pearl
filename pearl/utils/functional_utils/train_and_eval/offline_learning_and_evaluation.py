# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import io
import logging
import math
import os
import time

from typing import Optional

import torch

from pearl.api.environment import Environment
from pearl.pearl_agent import PearlAgent
from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.requests_get import requests_get
from pearl.utils.functional_utils.train_and_eval.learning_logger import (
    LearningLogger,
    null_learning_logger,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


logger = logging.Logger(__name__)


def is_file_readable(file_path: str) -> bool:
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def get_offline_data_in_buffer(
    is_action_continuous: bool,
    url: str | None = None,
    data_path: str | None = None,
    max_number_actions_if_discrete: int | None = None,
    size: int = 1000000,
    device: str = "cpu",
) -> ReplayBuffer:
    """
    Fetches offline data from a url and returns a replay buffer which can be sampled
    to train the offline agent. For this implementation we use BasicReplayBuffer.

    - Assumes the offline data is an iterable consisting of transition tuples
        (observation, action, reward, next_observation, curr_available_actions,
        next_available_actions, terminated, truncated) as dictionaries.

    - Also assumes offline data is in a .pt file; reading from a
        csv file can also be added later.

    Args:
        is_action_continuous (bool): whether the action space is continuous or discrete.
            for continuous actions spaces, we need to set this flag; see 'push' method
            in BasicReplayBuffer class
        url (str, optional): from where offline data needs to be fetched from
        data_path (str, optional): local path to the offline data
        max_number_actions_if_discrete (int, optional): To work with a discrete action space in
            Pearl, each transition tuple requires specifying the maximum number of actions in the
            action space. For offline learning, we expect users to know the maximum number of
            actions in the discrete action space.
            For continuous action spaces, we do not need to specify this parameter. Hence it is
            optional and defaults to None.
        size (int): Size of the replay buffer. Defaults to 1000000.
        device (cpu): Device to load the data onto. If no device is specified, defaults to cpu.

    Returns:
        ReplayBuffer: a BasicReplayBuffer containing offline data of transition tuples.
            The transition tuples are in the format as expected by a Pearl agent.
    """

    if is_action_continuous:
        if max_number_actions_if_discrete is not None:
            raise ValueError(
                "get_offline_data: is_action_continuous = True requires \
                max_number_actions to be None"
            )
    elif max_number_actions_if_discrete is None:
        raise ValueError(
            "get_offline_data: is_action_continuous = False requires \
            max_number_actions to be an integer value"
        )

    if url is not None:
        offline_transitions_data = requests_get(url)
        stream = io.BytesIO(offline_transitions_data.content)  # implements seek()
        raw_transitions_buffer = torch.load(
            stream, map_location=torch.device(device), weights_only=False
        )
    else:
        if data_path is None:
            raise ValueError(
                "provide either a data_path or a url to fetch offline data"
            )

        # loads data on the specified device
        raw_transitions_buffer = torch.load(
            data_path, map_location=torch.device(device), weights_only=False
        )

    offline_data_replay_buffer = BasicReplayBuffer(size)
    if is_action_continuous:
        offline_data_replay_buffer._is_action_continuous = True

    for transition in raw_transitions_buffer:
        if transition["curr_available_actions"].__class__.__name__ == "Discrete":
            transition["curr_available_actions"] = DiscreteActionSpace(
                actions=list(
                    torch.arange(transition["curr_available_actions"].n).view(-1, 1)
                )
            )
        if transition["next_available_actions"].__class__.__name__ == "Discrete":
            transition["next_available_actions"] = DiscreteActionSpace(
                actions=list(
                    torch.arange(transition["next_available_actions"].n).view(-1, 1)
                )
            )

        offline_data_replay_buffer.push(
            state=transition["observation"],
            action=transition["action"],
            reward=transition["reward"],
            next_state=transition["next_observation"],
            curr_available_actions=transition["curr_available_actions"],
            next_available_actions=transition["next_available_actions"],
            terminated=transition["done"],
            truncated=False,
            max_number_actions=max_number_actions_if_discrete,
        )

    return offline_data_replay_buffer


def offline_learning(
    offline_agent: PearlAgent,
    data_buffer: ReplayBuffer,
    training_epochs: Optional[float] = None,
    number_of_batches: Optional[int] = None,
    learning_logger: LearningLogger = null_learning_logger,
    seed: Optional[int] = None,
) -> None:
    """
    Trains the offline agent using transition tuples from offline data (provided in
    `data_buffer`). Must provide a replay buffer with transition tuples.
    You may choose to use `get_offline_data_in_buffer` to create an offline data buffer.

    The method calls `offline_agent.learn_batch` on `number_of_batches` batches of
    data from the replay buffer.
    If `number_of_batches` is not provided, then`training_epochs`
    is used to determine the number of batches to sample,
    with `len(data_buffer) / offline_agent.policy_learner.batch_size` batches sampled per epoch.

    Fractional numbers of epochs *are* allowed.
    If the number of batches computed from fractional epochs is also fractions, it is rounded up.

    If neither `number_of_batches` and `training_epochs` are provided,
    1 training epoch is used.
    If both `number_of_batches` and `training_epochs` are provided,
    an error is raise.

    The learning logger is invoked with parameters (loss, batch_index, batch)
    after each call to `offline_agent.learn_batch`,
    where `batch_index` is (i - 1) when `learn_batch` is called on the i-th sampled batch.

    Args:
        offline agent (PearAgent): a Pearl agent (typically a conservative one such as CQL or IQL).
        data_buffer (ReplayBuffer): a replay buffer to sample a batch of transition data.
        training_epochs (Optional[float], default 1): number of passes over training data.
                        Fractional values result in a rounded up number of samples batches.
                        Mutually exclusive with number_of_batches.
        number_of_batches (Optional[int], default 1000): number of batches
                                         (of size `offline_agent.policy_learner.batch_size`)
                                         to sample from the replay buffer.
                                         Mutually exclusive with training_epochs.
        logger (LearningLogger, optional): a LearningLogger to log the training loss
                                           (default is no-op logger).
        seed (int, optional): random seed (default is `int(time.time())`).
    """
    if seed is None:
        seed = int(time.time())
    set_seed(seed=seed)

    assert (
        len(data_buffer) > 0
    ), "offline_learning: data_buffer must have at least one transition tuple"

    effective_batch_size = min(
        offline_agent.policy_learner.batch_size, len(data_buffer)
    )

    if number_of_batches is None:
        if training_epochs is None:
            training_epochs = 1
        number_of_batches = math.ceil(
            training_epochs * len(data_buffer) / effective_batch_size
        )
    elif training_epochs is not None:
        raise ValueError(
            f"{offline_learning.__name__} must receive at most one of number_of_batches and "
            + "training_epochs, but got both."
        )

    logger.info(
        f"Training offline agent for {training_epochs} epochs, "
        f"policy learner batch size {offline_agent.policy_learner.batch_size}, "
        f"replay buffer size {len(data_buffer)}, "
        f"effective batch size {effective_batch_size}, and {number_of_batches} batches."
    )

    # move replay buffer to device of the offline agent
    data_buffer.device_for_batches = offline_agent.device

    # training loop
    for i in range(number_of_batches):
        batch = data_buffer.sample(effective_batch_size)
        assert isinstance(batch, TransitionBatch)
        loss = offline_agent.learn_batch(batch=batch)
        learning_logger(loss, i, batch)


def offline_evaluation(
    offline_agent: PearlAgent,
    env: Environment,
    number_of_episodes: int = 1000,
    seed: int | None = None,
) -> list[float]:
    """
    Evaluates the performance of an offline trained agent.

    Args:
        agent: the offline trained agent.
        env: the environment to evaluate the agent in
        number_of_episodes: the number of episodes to evaluate for.
    Returns:
        returns_offline_agent: a list of returns for each evaluation episode.
    """

    # sanity check: during offline evaluation, the agent should not learn or explore.
    learn = False
    exploit = True
    learn_after_episode = False

    returns_offline_agent = []
    for i in range(number_of_episodes):
        evaluation_seed = seed + i if seed is not None else None
        episode_info, total_steps = run_episode(
            agent=offline_agent,
            env=env,
            learn=learn,
            exploit=exploit,
            learn_after_episode=learn_after_episode,
            seed=evaluation_seed,
        )
        g = episode_info["return"]
        if i % 1 == 0:
            print(f"\repisode {i}, return={g}", end="")
        returns_offline_agent.append(g)

    return returns_offline_agent
