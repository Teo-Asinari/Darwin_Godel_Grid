import numpy as np

class GridWorld:
    def __init__(self, grid):
        self.grid = grid
        self.start_pos = tuple(map(int, np.argwhere(grid == 'S')[0]))
        self.goal_pos = tuple(map(int, np.argwhere(grid == 'G')[0]))
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = moves[action]

        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if (0 <= new_pos[0] < self.grid.shape[0] and
            0 <= new_pos[1] < self.grid.shape[1] and
            self.grid[new_pos] != '#'):
            self.agent_pos = new_pos

        done = self.agent_pos == self.goal_pos
        reward = 1 if done else -0.01  # tiny penalty per step
        return self.agent_pos, reward, done

    def render(self):
        grid_copy = self.grid.copy()
        r, c = self.agent_pos
        grid_copy[r, c] = 'A'
        for row in grid_copy:
            print(' '.join(row))
        print()

# Example usage:
if __name__ == "__main__":
    maze = np.array([
        list("S  #  "),
        list(" ## #G"),
        list("     ")
    ])
    env = GridWorld(maze)
    env.render()
