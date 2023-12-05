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
    info: Dict[str, Any]
    cost: Optional[float] = None
    available_action_space: Optional[ActionSpace] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
