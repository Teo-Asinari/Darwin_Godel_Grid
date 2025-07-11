import numpy as np
from typing import Tuple

class GridWorld:
    def __init__(self, grid: np.ndarray) -> None:
        self.grid: np.ndarray = grid

        if self.grid.dtype.kind != 'U' and self.grid.dtype.kind != 'S':
            raise ValueError("Grid dtype must be string (unicode or bytes)")

        start_matches = np.argwhere(self.grid == 'S')
        if start_matches.size == 0:
            raise ValueError(f"No 'S' found — unique values: {np.unique(self.grid)}")
        start_indices = start_matches[0]
        self.start_pos: Tuple[int, int] = (int(start_indices[0]), int(start_indices[1]))

        goal_matches = np.argwhere(self.grid == 'G')
        if goal_matches.size == 0:
            raise ValueError(f"No 'G' found — unique values: {np.unique(self.grid)}")
        goal_indices = goal_matches[0]
        self.goal_pos: Tuple[int, int] = (int(goal_indices[0]), int(goal_indices[1]))

        self.agent_pos: Tuple[int, int] = self.start_pos

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = moves[action]

        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if (0 <= new_pos[0] < self.grid.shape[0] and
            0 <= new_pos[1] < self.grid.shape[1] and
            self.grid[new_pos] != '#'):
            self.agent_pos = new_pos

        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01
        return self.agent_pos, reward, done

    def render(self) -> None:
        grid_copy = self.grid.copy()
        r, c = self.agent_pos
        grid_copy[r, c] = 'A'
        for row in grid_copy:
            print(' '.join(row))
        print()
