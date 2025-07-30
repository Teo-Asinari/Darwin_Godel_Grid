# dqn_model.py
import torch
import torch.nn as nn
from typing import Union

class DQN(nn.Module):
  def __init__(self, input_dim: int, num_actions: int) -> None:
    super(DQN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, num_actions)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
