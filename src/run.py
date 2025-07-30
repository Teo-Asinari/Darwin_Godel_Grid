from gridworld import GridWorld, Position, StepOutput
from agent import Agent
from q_learning.dqn_agent import DQNAgent
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

agent = DQNAgent(state_dim=2, num_actions=4)

episodes: int = 150

path = astar(env.grid, env.start_pos, env.goal_pos)
print("A* path: ", path)

for ep in range(episodes):
    state: Position = env.reset()
    done: bool = False
    total_reward: float = 0.0

    while not done:
        action: int = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store((state, action, reward, next_state, float(done)))
        agent.update(batch_size=32)
        state = next_state
        total_reward += reward

        if ep % 10 == 0:
            agent.update_target()
    agent.decay_epsilon()

    # Optional: render occasionally
    if ep % 50 == 0:
        env.render()
    print(f"Episode {ep+1} total reward: {total_reward}\n")
    agent.render_policy(env)