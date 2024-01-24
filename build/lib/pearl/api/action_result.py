# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pearl.api.action_space import ActionSpace

from pearl.api.observation import Observation

from pearl.api.reward import Reward


@dataclass
class ActionResult:
    observation: Observation
    reward: Reward
    terminated: bool
    truncated: bool
    info: Optional[Dict[str, Any]] = None
    cost: Optional[float] = None
    available_action_space: Optional[ActionSpace] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
