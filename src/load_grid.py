import numpy as np

def load_grid(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [list(line.strip('\n')) for line in lines]
    return np.array(lines)