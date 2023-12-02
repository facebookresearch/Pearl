import io
import logging
import os
import sys
from typing import List, Optional

import requests
import torch

try:
    from libfb.py.certpathpicker.cert_path_picker import get_client_credential_paths
except ImportError:
    print(
        "Cert path picker failed to find FB proxy certificates. Downloading data from "
        "a url directly won't work inside meta"
    )

from pearl.api.agent import Agent
from pearl.api.environment import Environment
from pearl.pearl_agent import PearlAgent
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

FWDPROXY_PORT = 8082
FWDPROXY_HOSTNAME = "https://fwdproxy"
FWDPROXY_CORP_HOSTNAME = "https://fwdproxy-regional-corp.{0}.fbinfra.net"
EXTERNAL_ENDPOINT = "https://www.google.com"
CORP_ENDPOINT = "https://npm.thefacebook.com/"

FB_CA_BUNDLE = "/var/facebook/rootcanal/ca.pem"

# pyre-fixme[5]: Global expression must be annotated.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def is_file_readable(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def get_offline_data_in_buffer(
    is_action_continuous: bool,
    url: Optional[str] = None,
    data_path: Optional[str] = None,
    size: int = 1000000,
) -> ReplayBuffer:
    """
    Fetches offline data from a url and returns a replay buffer which can be sampled
    to train the offline agent. For this implementation we use FIFOOffPolicyReplayBuffer.

    - Assumes the offline data is an iterable consisting of transition tuples
        (observation, action, reward, next_observation, curr_available_actions,
        next_available_actions, action_space_done) as dictionaries.

    - Also assumes offline data is in a .pt file; reading from a
        csv file can also be added later.

    Args:
        is_action_continuous: whether the action space is continuous or discrete.
            for continuous actions spaces, we need to set this flag; see 'push' method
            in FIFOOffPolicyReplayBuffer class
        url: from where offline data needs to be fetched from
        data_path: local path to the offline data
        size: size of the replay buffer

    Returns:
        ReplayBuffer: a FIFOOffPolicyReplayBuffer containing offline data of transition tuples.
    """
    if url is not None:
        thrift_cert, thrift_key = get_client_credential_paths()
        if not is_file_readable(thrift_cert) or not is_file_readable(thrift_key):
            raise RuntimeError("Missing key TLS cert settings.")

        fwdproxy_url = f"{FWDPROXY_HOSTNAME}:{FWDPROXY_PORT}"
        proxies = {"http": fwdproxy_url, "https": fwdproxy_url}
        client_cert = (thrift_cert, thrift_key)

        offline_transitions_data = requests.get(
            url, proxies=proxies, verify=FB_CA_BUNDLE, cert=client_cert
        )
        stream = io.BytesIO(offline_transitions_data.content)  # implements seek()
        raw_transitions_buffer = torch.load(stream)
    else:
        raw_transitions_buffer = torch.load(data_path)  # pyre-ignore

    offline_data_replay_buffer = FIFOOffPolicyReplayBuffer(size)
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
        if transition["action_space"].__class__.__name__ == "Discrete":
            transition["action_space"] = DiscreteActionSpace(
                actions=list(torch.arange(transition["action_space"].n).view(-1, 1))
            )

        offline_data_replay_buffer.push(
            transition["observation"],
            transition["action"],
            transition["reward"],
            transition["next_observation"],
            transition["curr_available_actions"],
            transition["next_available_actions"],
            transition["action_space"],
            transition["done"],
        )

    return offline_data_replay_buffer


def offline_learning(
    offline_agent: PearlAgent,
    data_buffer: ReplayBuffer,
    training_epochs: int = 1000,
    seed: int = 100,
) -> None:
    """
    Trains the offline agent using transition tuples from offline data (provided in
    the data_buffer). Must provide a replay buffer with transition tuples - please
    use the method get_offline_data_in_buffer to create an offline data buffer.

    Args:
        offline agent: a conservative learning agent (CQL or IQL).
        data_buffer: a replay buffer to sample a batch of transition data.
        training_epochs: number of training epochs for offline learning.
    """
    set_seed(seed=seed)

    # move replay buffer to device of the offline agent
    data_buffer.device = offline_agent.device

    # training loop
    for i in range(training_epochs):
        batch = data_buffer.sample(offline_agent.policy_learner.batch_size)
        assert isinstance(batch, TransitionBatch)
        loss = offline_agent.learn_batch(batch=batch)
        if i % 500 == 0:
            print("training epoch", i, "training loss", loss)


def offline_evaluation(
    offline_agent: Agent,
    env: Environment,
    number_of_episodes: int = 1000,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Evaluates the performance of an offline trained agent.

    Args:
        agent: the offline trained agent.
        env: the environment to evaluate the agent in
        number_of_episodes: the number of episodes to evaluate for.
    Returns:
        returns_offline_agent: a list of returns for each evaluation episode.
    """

    # check: during offline evaluation, the agent should not learn or explore.
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
