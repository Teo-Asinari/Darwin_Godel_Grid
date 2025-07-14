import heapq


# An interesting primer on A*
# https://www.youtube.com/watch?v=ySN5Wnu88nE
# https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html
# A* is a small extension to Dijkstra's algorithm (flood-fill style) which
# adds an extra heuristic term to the cost function. The heuristic is usually
# based on some distance metric and assumes knowledge of the start and goal coordinates.

def heuristic(a, b):
  return abs(a[0] - b[0]) + abs(a[1] - b[1])
  
def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    return path[::-1]


def astar(grid, start, goal):
  rows, cols = grid.shape
  open_set = []
  heapq.heappush(open_set, (0, start)) # heapq uses the first element of the tuple to sort the elements
  came_from = {}
  g_score = {start: 0}

  while open_set:
    _, current = heapq.heappop(open_set)

    if current == goal:
      return reconstruct_path(came_from, current)
    
    for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      neighbor = (current[0] + move[0], current[1] + move[1])

      if (0 <= neighbor[0] < rows and 
          0 <= neighbor[1] < cols and
          grid[neighbor] != '#'):

        tentative_g_score = g_score[current] + 1

        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
          came_from[neighbor] = current
          g_score[neighbor] = tentative_g_score
          f_score = tentative_g_score + heuristic(neighbor, goal)
          heapq.heappush(open_set, (f_score, neighbor))

  return None