from typing import List, Tuple, Dict, Optional, Set
import heapq
import numpy as np


# An interesting primer on A*
# https://www.youtube.com/watch?v=ySN5Wnu88nE
# https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html
# A* is a small extension to Dijkstra's algorithm (flood-fill style) which
# adds an extra heuristic term to the cost function. The heuristic is usually
# based on some distance metric and assumes knowledge of the start and goal coordinates.

# Type aliases for better readability
Position = Tuple[int, int]
Grid = np.ndarray
Path = List[Position]
CameFrom = Dict[Position, Position]
GScore = Dict[Position, float]
OpenSetItem = Tuple[float, Position]


def heuristic(a: Position, b: Position) -> float:
  return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: CameFrom, current: Position) -> Path:
  path: Path = []
  while current in came_from:
    path.append(current)
    current = came_from[current]
  return path[::-1]


def astar(grid: Grid, start: Position, goal: Position) -> Optional[Path]:
  rows: int
  cols: int
  rows, cols = grid.shape
  
  open_set: List[OpenSetItem] = []
  heapq.heappush(open_set, (0, start))
  
  came_from: CameFrom = {}
  g_score: GScore = {start: 0}
  
  # Set to track positions in open set for O(1) lookup
  open_set_positions: Set[Position] = {start}

  while open_set:
    _, current = heapq.heappop(open_set)
    open_set_positions.remove(current)

    if current == goal:
      return reconstruct_path(came_from, current)
    
    moves: List[Position] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for move in moves:
      neighbor: Position = (current[0] + move[0], current[1] + move[1])

      # Check if neighbor is within bounds and not a wall
      if (0 <= neighbor[0] < rows and 
          0 <= neighbor[1] < cols and
          grid[neighbor] != '#'):
        
        tentative_g_score: float = g_score[current] + 1

        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
          came_from[neighbor] = current
          g_score[neighbor] = tentative_g_score
          f_score: float = tentative_g_score + heuristic(neighbor, goal)
          
          # Only add to open set if not already there
          if neighbor not in open_set_positions:
            heapq.heappush(open_set, (f_score, neighbor))
            open_set_positions.add(neighbor)

  return None