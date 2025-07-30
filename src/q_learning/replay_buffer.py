# replay_buffer.py
import random
from typing import List, Tuple, Any

# Type alias for transitions
Transition = Tuple[Any, ...]

class ReplayBuffer:
  def __init__(self, capacity: int) -> None:
    self.buffer: List[Transition] = []
    self.capacity: int = capacity

  def push(self, transition: Transition) -> None:
    self.buffer.append(transition)
    if len(self.buffer) > self.capacity:
      self.buffer.pop(0)

  def sample(self, batch_size: int) -> List[Transition]:
    return random.sample(self.buffer, batch_size)

  def __len__(self) -> int:
    return len(self.buffer)
