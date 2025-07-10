import numpy as np
from typing import Optional, Any

class Agent:
    def __init__(self, policy: Optional[Any] = None) -> None:
        self.policy = policy

    def act(self, state: Any) -> int:
        return int(np.random.choice([0, 1, 2, 3]))

    def mutate(self) -> "Agent":
        return Agent(policy=self.policy)