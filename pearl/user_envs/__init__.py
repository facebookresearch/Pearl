# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from .envs import *  # noqa
from .wrappers import *  # noqa

try:
    from gymnasium.envs.registration import register
except ModuleNotFoundError:
    from gym.envs.registration import register

for game in ["Catcher", "FlappyBird", "Pixelcopter", "PuckWorld", "Pong"]:
    register(id="{}-PLE-v0".format(game), entry_point=f"gym_pygame.envs:{game}Env")


register(
    id="MeanVarBandit-v0",
    entry_point="pearl.user_envs.envs:MeanVarBanditEnv",
)

register(
    id="Catcher-PLE-500-v0",
    entry_point="gym_pygame.envs:CatcherEnv",
    max_episode_steps=500,
)

register(
    id="FlappyBird-PLE-500-v0",
    entry_point="gym_pygame.envs:FlappyBirdEnv",
    max_episode_steps=500,
)

register(
    id="Pong-PLE-500-v0",
    entry_point="gym_pygame.envs:PongEnv",
    max_episode_steps=500,
)

register(
    id="Pixelcopter-PLE-500-v0",
    entry_point="gym_pygame.envs:PixelcopterEnv",
    max_episode_steps=500,
)

register(
    id="PuckWorld-PLE-500-v0",
    entry_point="gym_pygame.envs:PuckWorldEnv",
    max_episode_steps=500,
)
