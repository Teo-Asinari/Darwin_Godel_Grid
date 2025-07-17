from gridworld import GridWorld
from agent import Agent
from q_learning.tabular_qlearn_agent import TabularQLearnAgent
import numpy as np
from astar import astar
from load_grid import load_grid

maze = load_grid("src/gridworlds/gridworldMedium.txt")
# maze = np.array([
#     list("S  #  "),
#     list(" ## #G"),
#     list("      ")
# ])

env = GridWorld(maze)

agent = TabularQLearnAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

episodes = 150

path = astar(env.grid, env.start_pos, env.goal_pos)
print("A* path: ", path)

for ep in range(episodes):
    state = env.reset()
    done: bool = False
    total_reward: float = 0.0

    while not done:
        action = agent.act(state)                      # pick action
        next_state, reward, done = env.step(action)    # take action
        agent.learn(state, action, reward, next_state) # update Q-table
        state = next_state                             # advance
        total_reward += reward

        # Optional: render occasionally
        if ep % 50 == 0:
            env.render()
    print(f"Episode {ep+1} total reward: {total_reward}\n")
    agent.render_policy(env)