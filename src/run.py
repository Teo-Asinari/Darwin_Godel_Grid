from gridworld import GridWorld
from agent import Agent
import numpy as np
from astar import astar

maze = np.array([
    list("S  #  "),
    list(" ## #G"),
    list("      ")
])

env = GridWorld(maze)

agent = Agent()

episodes = 5

for ep in range(episodes):
    state = env.reset()
    done: bool = False
    total_reward: float = 0.0

    path = astar(env.grid, env.start_pos, env.goal_pos)
    print("A* path: ", path)

    # while not done:
    #     action: int = agent.act(state)
    #     state, reward, done = env.step(action)
    #     total_reward += reward
    #     env.render()

    print(f"Episode {ep+1} total reward: {total_reward}\n")
