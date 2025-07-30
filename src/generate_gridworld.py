import numpy as np
import os
import time
from typing import List, Tuple

def generate_gridworld(rows: int, cols: int, wall_prob: float = 0.2, ensure_path: bool = True) -> np.ndarray:
    """
    Generate a GridWorld as a numpy array with boundary walls, start (S), goal (G),
    and (optionally) a guaranteed path from S to G.
    """
    grid = np.full((rows, cols), '#')
    grid[1:-1, 1:-1] = ' '  # Open interior

    # Place start
    grid[1, 1] = 'S'

    # Optionally carve a guaranteed path from S to G
    if ensure_path:
        r, c = 1, 1
        while (r, c) != (rows-2, cols-2):
            if r < rows-2 and (c == cols-2 or np.random.rand() < 0.5):
                r += 1
            elif c < cols-2:
                c += 1
            grid[r, c] = ' '  # Carve path

    # Add random interior walls, but don't overwrite S or the path
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r, c] == ' ' and (r, c) not in [(1, 1), (rows-2, cols-2)]:
                if np.random.rand() < wall_prob:
                    grid[r, c] = '#'

    # Place goal after all other modifications
    grid[rows-2, cols-2] = 'G'

    return grid

def save_gridworld_to_txt(grid: np.ndarray, filename: str) -> None:
    """
    Save the gridworld numpy array to a text file, one row per line, no delimiters.
    """
    with open(filename, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')

class GridWorldBatchGenerator:
    def __init__(self, output_dir: str = "src/gridworlds") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_and_save_batch(self, sizes: List[int], num_per_size: int = 10, wall_prob: float = 0.2) -> None:
        for size in sizes:
            rows, cols = size, size
            for i in range(num_per_size):
                grid = generate_gridworld(rows, cols, wall_prob=wall_prob, ensure_path=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                filename = f"gridworld_{rows}x{cols}_{i+1}_{timestamp}.txt"
                filepath = os.path.join(self.output_dir, filename)
                save_gridworld_to_txt(grid, filepath)
                print(f"Saved: {filepath}")

if __name__ == "__main__":
    # Example: generate 10 gridworlds each of sizes 10x10, 50x50, 100x100, 150x150
    batch_gen = GridWorldBatchGenerator(output_dir="src/gridworlds")
    sizes: List[int] = [10, 50, 100, 150]
    batch_gen.generate_and_save_batch(sizes, num_per_size=10, wall_prob=0.2) 