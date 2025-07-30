import numpy as np
from typing import Any, Tuple, Dict, List
from agent import Agent  # your base class
from gridworld import Position

# Type aliases for better readability
State = Position  # Alias for consistency
QTable = Dict[Tuple[State, int], float]

class TabularQLearnAgent(Agent):
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2) -> None:
        super().__init__(policy=None)
        self.alpha: float = alpha      # Learning rate
        self.gamma: float = gamma      # Discount factor
        self.epsilon: float = epsilon  # Exploration probability
        self.q_table: QTable = {}       # (state, action) -> Q-value

    def get_q(self, state: State, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def act(self, state: State) -> int:
        # ε-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        else:
            qs: List[float] = [self.get_q(state, a) for a in range(4)]
            max_q: float = max(qs)
            best_actions: List[int] = [a for a, q in enumerate(qs) if q == max_q]
            return np.random.choice(best_actions)

    def learn(self, state: State, action: int, reward: float, next_state: State) -> None:
        old_q: float = self.get_q(state, action)
        next_qs: List[float] = [self.get_q(next_state, a) for a in range(4)]
        td_target: float = reward + self.gamma * max(next_qs)
        td_error: float = td_target - old_q
        new_q: float = old_q + self.alpha * td_error
        self.q_table[(state, action)] = new_q
    
    def render_policy(self, env) -> None:
        action_symbols: List[str] = ['↑', '↓', '←', '→']
        policy_grid = np.full(env.grid.shape, ' ')

        for r in range(env.grid.shape[0]):
            for c in range(env.grid.shape[1]):
                if env.grid[r, c] == '#' or env.grid[r, c] == 'G':
                    continue
                state: State = (r, c)
                qs: List[float] = [self.get_q(state, a) for a in range(4)]
                best_action: int = np.argmax(qs)
                policy_grid[r, c] = action_symbols[best_action]

        print("\nLearned Policy:")
        for row in policy_grid:
            print(' '.join(row))

