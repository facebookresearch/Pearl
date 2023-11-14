import logging
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from pearl.api.agent import Agent
from pearl.api.environment import Environment
from pearl.api.reward import Value
from pearl.utils.functional_utils.experimentation.plots import fontsize_for

MA_WINDOW_SIZE = 10


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def latest_moving_average(data):
    return (
        sum(data[-MA_WINDOW_SIZE:]) * 1.0 / MA_WINDOW_SIZE
        if len(data) >= MA_WINDOW_SIZE
        else sum(data) * 1.0 / len(data)
    )


def online_learning_to_png_graph(
    agent: Agent,
    env: Environment,
    # pyre-fixme[2]: Parameter must be annotated.
    filename="returns.png",
    # pyre-fixme[2]: Parameter must be annotated.
    number_of_episodes=1000,
    # pyre-fixme[2]: Parameter must be annotated.
    learn_after_episode=False,
) -> None:
    """
    Runs online learning and generates a PNG graph of the returns.

    Args:
        agent (Agent): the agent.
        env (Environment): the environment.
        filename (str, optional): the filename to save to. Defaults to "returns.png".
        number_of_episodes (int, optional): the number of episodes to run. Defaults to 1000.
        learn_after_episode: if we want to learn after episode or learn every step during episode
    """

    returns = online_learning_returns(
        agent=agent,
        env=env,
        number_of_episodes=number_of_episodes,
        learn_after_episode=learn_after_episode,
    )

    if filename is not None:
        title = f"{str(agent)} on {str(env)}"
        if len(title) > 125:
            logging.warning(
                f"Figure title is very long, with {len(title)} characters: {title}"
            )
        plt.plot(returns)
        plt.title(title, fontsize=fontsize_for(title))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.savefig(filename)
        plt.close()


def online_learning_returns(
    agent: Agent,
    env: Environment,
    number_of_episodes: int = 1000,
    number_of_steps: Optional[int] = None,
    learn_after_episode: bool = False,
    print_every_x_episodes: Optional[int] = None,
    print_every_x_steps: Optional[int] = None,
) -> List[Value]:
    returns = []
    online_learning(
        agent,
        env,
        number_of_episodes=number_of_episodes,
        number_of_steps=number_of_steps,
        learn_after_episode=learn_after_episode,
        process_return=returns.append,
        print_every_x_episodes=print_every_x_episodes,
        print_every_x_steps=print_every_x_steps,
    )
    return returns


# pyre-fixme[3]: Return type must be annotated.
def online_learning(
    agent: Agent,
    env: Environment,
    number_of_episodes: int = 1000,
    number_of_steps: Optional[int] = None,
    learn_after_episode: bool = False,
    process_return: Callable[[Value], None] = lambda g: None,
    print_every_x_episodes: Optional[int] = None,
    print_every_x_steps: Optional[int] = None,
):
    """
    Performs online learning for a number of episodes.

    Args:
        agent (Agent): the agent.
        env (Environment): the environmnent.
        number_of_episodes (int, optional): the number of episodes to run. Defaults to 1000.
        learn_after_episode (bool, optional): asks the agent to only learn after every episode. Defaults to False.
        process_return (Callable[[Value], None], optional): a callable for processing the returns of the episodes. Defaults to no-op.
    """
    assert (number_of_episodes is None and number_of_steps is not None) or (
        number_of_episodes is not None and number_of_steps is None
    )
    total_steps = 0
    total_episodes = 0
    returns_within_period = []
    while True:
        if number_of_episodes is not None and total_episodes >= number_of_episodes:
            break
        if number_of_steps is not None and total_steps >= number_of_steps:
            break
        old_total_steps = total_steps
        g, total_steps = episode_return(
            agent,
            env,
            learn=True,
            exploit=False,
            learn_after_episode=learn_after_episode,
            total_steps=old_total_steps,
        )
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
                f"\repisode {total_episodes}, step {total_steps}, agent={agent}, env={env}, return={g}",  # noqa E501
                end="",
            )
        if number_of_episodes is not None:
            process_return(g)
        if number_of_steps is not None:
            returns_within_period.append(g)
            if old_total_steps // (number_of_steps // 100) < (total_steps) // (
                number_of_steps // 100
            ):  # record the average of returns every 1% of steps
                process_return(np.mean(returns_within_period))
                returns_within_period = []


def target_return_is_reached(
    target_return: Value,
    max_episodes: int,
    agent: Agent,
    env: Environment,
    learn: bool,
    learn_after_episode: bool,
    exploit: bool,
    # pyre-fixme[2]: Parameter must be annotated.
    required_target_returns_in_a_row=1,
    check_moving_average: bool = False,
) -> bool:
    """
    Learns until obtaining target return (a certain number of times in a row, default 1) or max_episodes are completed.
    Args
        target_return (Value): the targeted return.
        max_episodes (int): the maximum number of episodes to run.
        agent (Agent): the agent.
        env (Environment): the environment.
        learn (bool): whether to learn.
        learn_after_episode (bool): whether to learn after every episode.
        exploit (bool): whether to exploit.
        required_target_returns_in_a_row (int, optional): how many times we must hit the target to succeed.
        check_moving_average: if this is enabled, we check the if latest moving average value reaches goal
    Returns
        bool: whether target_return has been obtained required_target_returns_in_a_row times in a row.
    """
    target_returns_in_a_row = 0
    returns = []
    total_steps = 0
    for i in range(max_episodes):
        if i % 10 == 0:
            print(f"episode {i}")
        g, total_steps = episode_return(
            agent=agent,
            env=env,
            learn=learn,
            learn_after_episode=learn_after_episode,
            exploit=exploit,
            total_steps=total_steps,
        )
        returns.append(g)
        value = g if not check_moving_average else latest_moving_average(returns)
        if value >= target_return:
            target_returns_in_a_row += 1
            if target_returns_in_a_row >= required_target_returns_in_a_row:
                return True
        else:
            target_returns_in_a_row = 0
    return False


# pyre-fixme[3]: Return type must be annotated.
def episode_return(
    agent: Agent,
    env: Environment,
    learn: bool = False,
    exploit: bool = True,
    learn_after_episode: bool = False,
    total_steps: int = 0,
):
    """
    Runs one episode and returns the total reward (return).

    Args:
        agent (Agent): the agent.
        env (Environment): the environment.
        learn (bool, optional): Runs `agent.learn()` after every step. Defaults to False.
        exploit (bool, optional): asks the agent to only exploit. Defaults to False.
        learn_after_episode (bool, optional): asks the agent to only learn after every episode. Defaults to False.

    Returns:
        Value: the return of the episode.
    """
    g = 0
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    done = False
    while not done:
        # pyre-fixme[28]: Unexpected keyword argument `exploit`.
        action = agent.act(exploit=exploit)
        action = (
            action.cpu() if isinstance(action, torch.Tensor) else action
        )  # action can be int sometimes
        action_result = env.step(action)
        g += action_result.reward
        agent.observe(action_result)
        if learn and not learn_after_episode:
            agent.learn()
        done = action_result.done
        total_steps += 1

    if learn and learn_after_episode:
        agent.learn()

    return g, total_steps
