import numpy as np
from typing import Any, Tuple
from agent import Agent  # your base class

class TabularQLearnAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        super().__init__(policy=None)
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.q_table = {}       # (state, action) -> Q-value

    def get_q(self, state: Tuple[int, int], action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def act(self, state: Tuple[int, int]) -> int:
        # ε-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        else:
            qs = [self.get_q(state, a) for a in range(4)]
            max_q = max(qs)
            best_actions = [a for a, q in enumerate(qs) if q == max_q]
            return np.random.choice(best_actions)

    def learn(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]) -> None:
        old_q = self.get_q(state, action)
        next_qs = [self.get_q(next_state, a) for a in range(4)]
        td_target = reward + self.gamma * max(next_qs)
        td_error = td_target - old_q
        new_q = old_q + self.alpha * td_error
        self.q_table[(state, action)] = new_q
