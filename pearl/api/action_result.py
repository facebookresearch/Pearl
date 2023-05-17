from dataclasses import dataclass
from typing import Dict

from pearl.api.observation import Observation

from pearl.api.reward import Reward


@dataclass
class ActionResult:
    observation: Observation
    reward: Reward
    terminated: bool
    truncated: bool
    info: Dict

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
