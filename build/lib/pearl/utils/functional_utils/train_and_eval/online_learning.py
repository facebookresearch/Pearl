# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pearl.api.environment import Environment
from pearl.api.reward import Value
from pearl.pearl_agent import PearlAgent
from pearl.utils.functional_utils.experimentation.plots import fontsize_for

MA_WINDOW_SIZE = 10


def latest_moving_average(data: List[Value]) -> float:
    return (
        sum(data[-MA_WINDOW_SIZE:]) * 1.0 / MA_WINDOW_SIZE  # pyre-ignore
        if len(data) >= MA_WINDOW_SIZE
        else sum(data) * 1.0 / len(data)  # pyre-ignore
    )


def online_learning_to_png_graph(
    agent: PearlAgent,
    env: Environment,
    filename: str = "returns.png",
    number_of_episodes: int = 1000,
    learn_after_episode: bool = False,
) -> None:
    """
    Runs online learning and generates a PNG graph of the returns.

    Args:
        agent (PearlAgent): the agent.
        env (Environment): the environment.
        filename (str, optional): the filename to save to. Defaults to "returns.png".
        number_of_episodes (int, optional): the number of episodes to run. Defaults to 1000.
        learn_after_episode: if we want to learn after episode or learn every step during episode
    """

    info = online_learning(
        agent=agent,
        env=env,
        number_of_episodes=number_of_episodes,
        learn_after_episode=learn_after_episode,
    )
    assert "return" in info

    if filename is not None:
        title = f"{str(agent)} on {str(env)}"
        if len(title) > 125:
            logging.warning(
                f"Figure title is very long, with {len(title)} characters: {title}"
            )
        plt.plot(info["return"])
        plt.title(title, fontsize=fontsize_for(title))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.savefig(filename)
        plt.close()


def online_learning(
    agent: PearlAgent,
    env: Environment,
    number_of_episodes: Optional[int] = None,
    number_of_steps: Optional[int] = None,
    learn_after_episode: bool = False,
    print_every_x_episodes: Optional[int] = None,
    print_every_x_steps: Optional[int] = None,
    seed: Optional[int] = None,
    # if number_of_episodes is used, report every record_period episodes
    # if number_of_steps is used, report every record_period steps
    # episodic stats collected within the period are averaged and then reported
    record_period: int = 1,
) -> Dict[str, Any]:
    """
    Performs online learning for a number of episodes.

    Args:
        agent (PearlAgent): the agent.
        env (Environment): the environmnent.
        number_of_episodes (int, optional): the number of episodes to run. Defaults to 1000.
        learn_after_episode (bool, optional): asks the agent to only learn after every episode.
        Defaults to False.
    """
    assert (number_of_episodes is None and number_of_steps is not None) or (
        number_of_episodes is not None and number_of_steps is None
    )
    total_steps = 0
    total_episodes = 0
    info = {}
    info_period = {}
    while True:
        if number_of_episodes is not None and total_episodes >= number_of_episodes:
            break
        if number_of_steps is not None and total_steps >= number_of_steps:
            break
        old_total_steps = total_steps
        episode_info, episode_total_steps = run_episode(
            agent,
            env,
            learn=True,
            exploit=False,
            learn_after_episode=learn_after_episode,
            total_steps=old_total_steps,
            seed=seed,
        )
        if number_of_steps is not None and episode_total_steps > record_period:
            print(
                f"An episode is longer than the report_period: episode length {episode_total_steps}"
                ", record_period {record_period}. Try using a smaller record_period."
            )
            exit(1)
        total_steps += episode_total_steps
        total_episodes += 1
        if (
            print_every_x_steps is not None
            and old_total_steps // print_every_x_steps
            < total_steps // print_every_x_steps
        ) or (
            print_every_x_episodes is not None
            and total_episodes % print_every_x_episodes == 0
        ):
            print(
                f"episode {total_episodes}, step {total_steps}, agent={agent}, env={env}",
            )
            for key in episode_info:
                print(f"{key}: {episode_info[key]}")
        for key in episode_info:
            info_period.setdefault(key, []).append(episode_info[key])
        if number_of_episodes is not None and (
            total_episodes % record_period == 0
        ):  # record average info value every report_period episodes
            for key in info_period:
                info.setdefault(key, []).append(np.mean(info_period[key]))
            info_period = {}
        if number_of_steps is not None and old_total_steps // record_period < (
            total_steps
        ) // (
            record_period
        ):  # record average info value every record_period steps
            for key in info_period:
                info.setdefault(key, []).append(np.mean(info_period[key]))
            info_period = {}
    return info


def target_return_is_reached(
    target_return: Value,
    max_episodes: int,
    agent: PearlAgent,
    env: Environment,
    learn: bool,
    learn_after_episode: bool,
    exploit: bool,
    required_target_returns_in_a_row: int = 1,
    check_moving_average: bool = False,
) -> bool:
    """
    Learns until obtaining target return (a certain number of times in a row, default 1)
    or max_episodes are completed.
    Args
        target_return (Value): the targeted return.
        max_episodes (int): the maximum number of episodes to run.
        agent (Agent): the agent.
        env (Environment): the environment.
        learn (bool): whether to learn.
        learn_after_episode (bool): whether to learn after every episode.
        exploit (bool): whether to exploit.
        required_target_returns_in_a_row (int, optional): how many times we must hit the target
        to succeed.
        check_moving_average: if this is enabled, we check the if latest moving average value
                              reaches goal
    Returns
        bool: whether target_return has been obtained required_target_returns_in_a_row times
              in a row.
    """
    target_returns_in_a_row = 0
    returns = []
    total_steps = 0
    for i in range(max_episodes):
        if i % 10 == 0 and i != 0:
            print(f"episode {i} return:", returns[-1])
        episode_info, episode_total_steps = run_episode(
            agent=agent,
            env=env,
            learn=learn,
            learn_after_episode=learn_after_episode,
            exploit=exploit,
        )
        total_steps += episode_total_steps
        returns.append(episode_info["return"])
        value = (
            episode_info["return"]
            if not check_moving_average
            else latest_moving_average(returns)
        )
        if value >= target_return:
            target_returns_in_a_row += 1
            if target_returns_in_a_row >= required_target_returns_in_a_row:
                return True
        else:
            target_returns_in_a_row = 0
    return False


def run_episode(
    agent: PearlAgent,
    env: Environment,
    learn: bool = False,
    exploit: bool = True,
    learn_after_episode: bool = False,
    total_steps: int = 0,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Runs one episode and returns an info dict and number of steps taken.

    Args:
        agent (Agent): the agent.
        env (Environment): the environment.
        learn (bool, optional): Runs `agent.learn()` after every step. Defaults to False.
        exploit (bool, optional): asks the agent to only exploit. Defaults to False.
        learn_after_episode (bool, optional): asks the agent to only learn at
                                              the end of the episode. Defaults to False.

    Returns:
        Tuple[Dict[str, Any], int]: the return of the episode and the number of steps taken.
    """
    if seed is None:
        observation, action_space = env.reset(seed=seed)
    else:
        # each episode has a different seed
        observation, action_space = env.reset(seed=seed + total_steps)
    agent.reset(observation, action_space)
    cum_reward = 0
    cum_cost = 0
    done = False
    episode_steps = 0
    num_risky_sa = 0
    while not done:
        action = agent.act(exploit=exploit)
        action = (
            action.cpu() if isinstance(action, torch.Tensor) else action
        )  # action can be int sometimes
        action_result = env.step(action)
        cum_reward += action_result.reward
        if (
            num_risky_sa is not None
            and action_result.info is not None
            and "risky_sa" in action_result.info
        ):
            num_risky_sa += action_result.info["risky_sa"]
        else:
            num_risky_sa = None
        if cum_cost is not None and action_result.cost is not None:
            cum_cost += action_result.cost
        else:
            cum_cost = None
        agent.observe(action_result)
        if learn and not learn_after_episode:
            agent.learn()
        done = action_result.done
        episode_steps += 1

    if learn and learn_after_episode:
        agent.learn()

    info = {"return": cum_reward}
    if num_risky_sa is not None:
        info.update({"risky_sa_ratio": num_risky_sa / episode_steps})
    if cum_cost is not None:
        info.update({"return_cost": cum_cost})

    return info, episode_steps
