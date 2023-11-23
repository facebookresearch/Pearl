from dataclasses import dataclass
from typing import Any, Dict, Optional

from pearl.api.observation import Observation

from pearl.api.reward import Reward


@dataclass
class ActionResult:
    observation: Observation
    reward: Reward
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    cost: Optional[Reward] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
