# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Union, Optional
from q_learning.dqn import DQN
from q_learning.replay_buffer import ReplayBuffer
from torch.optim.adam import Adam
from gridworld import Position

# Type aliases for better readability
State = Position  # Alias for consistency
Transition = Tuple[State, int, float, State, float]
Batch = List[Transition]

class DQNAgent:
    def __init__(self, state_dim: int, num_actions: int) -> None:
        self.q_net = DQN(state_dim, num_actions)
        self.target_net = DQN(state_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = Adam(self.q_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(capacity=5000)

        self.gamma: float = 0.99
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.05
        self.epsilon_decay: float = 0.995
        self.num_actions: int = num_actions

    def select_action(self, state: State) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def store(self, transition: Transition) -> None:
        self.buffer.push(transition)

    def update(self, batch_size: int) -> None:
        if len(self.buffer) < batch_size:
            return

        batch: Batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q
        q_values = self.q_net(states).gather(1, actions)

        # Target Q
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def render_policy(self, env) -> None:
        action_symbols: List[str] = ['↑', '↓', '←', '→']
        policy_grid = np.full(env.grid.shape, ' ')

        for r in range(env.grid.shape[0]):
            for c in range(env.grid.shape[1]):
                if env.grid[r, c] == '#' or env.grid[r, c] == 'G':
                    continue
                state: State = (r, c)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net(state_tensor)
                best_action = int(torch.argmax(q_values).item())
                policy_grid[r, c] = action_symbols[best_action]

        print("\nLearned Policy:")
        for row in policy_grid:
            print(' '.join(row))
