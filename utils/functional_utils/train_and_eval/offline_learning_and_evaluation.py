import io
import logging
import os
import sys

import requests
import torch
from libfb.py.certpathpicker.cert_path_picker import get_client_credential_paths
from pearl.api.agent import Agent
from pearl.api.environment import Environment
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import episode_return

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


def get_offline_data_in_buffer(url: str, size: int = 1000000) -> ReplayBuffer:
    """
    Fetches offline data from a url and returns a replay buffer which can be sampled
    to train the offline agent. For this implementation we use FIFOOffPolicyReplayBuffer.

    Args:
        url: from where offline data needs to be fetched from.

        - Assumes the offline data is an iterable consisting of transition tuples
        (observation, action, reward, next_observation, curr_available_actions,
        next_available_actions, action_space_done) as dictionaries.

        - Also assumes offline data is in a .pt file; reading from a
        csv file can also be added later.
    Returns:
        ReplayBuffer: a FIFOOffPolicyReplayBuffer containing offline data
        of transition tuples.
    """

    thrift_cert, thrift_key = get_client_credential_paths()
    if not is_file_readable(thrift_cert) or not is_file_readable(thrift_key):
        raise RuntimeError("Missing key TLS cert settings.")

    fwdproxy_url = f"{FWDPROXY_HOSTNAME}:{FWDPROXY_PORT}"
    proxies = {"http": fwdproxy_url, "https": fwdproxy_url}
    client_cert = (thrift_cert, thrift_key)

    offline_transitions_data = requests.get(
        url, proxies=proxies, verify=FB_CA_BUNDLE, cert=client_cert
    )

    offline_data_replay_buffer = FIFOOffPolicyReplayBuffer(size)
    stream = io.BytesIO(offline_transitions_data.content)  # implements seek()
    raw_transitions_buffer = torch.load(stream)
    for transition in raw_transitions_buffer:
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
    url: str,
    offline_agent: Agent,
    # pyre-fixme[9]: data_buffer has type `ReplayBuffer`; used as `None`.
    data_buffer: ReplayBuffer = None,
    training_epochs: int = 1000,
) -> None:
    """
    Trains the offline agent using transition tuples from offline data
    (provided in the data_buffer). Loads offline data from a url if
    data_buffer not provided.

    Args:
        offline agent: a conservative learning agent (CQL or IQL).
        data_buffer: a replay buffer to sample a batch of transition data.
        training_epochs: number of training epochs for offline learning.
    """
    set_seed(100)
    if data_buffer is None:
        # load data from a url and store it in a FIFOOffPolicyReplayBuffer
        data_buffer = get_offline_data_in_buffer(url)
        print("data buffer loaded")

    # training loop
    for i in range(training_epochs):
        # pyre-fixme[16]: `Agent` has no attribute `policy_learner`.
        batch = data_buffer.sample(offline_agent.policy_learner.batch_size)
        # pyre-fixme[16]: `Agent` has no attribute `learn_batch`.
        loss = offline_agent.learn_batch(batch=batch)
        if i % 500 == 0:
            print("training epoch", i, "training loss", loss)


# pyre-fixme[3]: Return type must be annotated.
def offline_evaluation(
    offline_agent: Agent,
    env: Environment,
    number_of_episodes: int = 1000,
):
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
        g = episode_return(
            agent=offline_agent,
            env=env,
            learn=learn,
            exploit=exploit,
            learn_after_episode=learn_after_episode,
        )
        if i % 100 == 0:
            print(f"\repisode {i}, return={g}", end="")
        returns_offline_agent.append(g)

    return returns_offline_agent
