import copy
import random
from q_learning.dqn_agent import DQNAgent
from gridworld import GridWorld
from load_grid import load_grid

# 1. Define your GridWorld suite
grid_files = [
    "src/gridworlds/gridworldMedium.txt",
    "src/gridworlds/gridworldVeryLarge.txt",
    # Add more grid files as needed
]
envs = [GridWorld(load_grid(f)) for f in grid_files]

# 2. Define initial agent config
def create_agent(config):
    return DQNAgent(state_dim=2, num_actions=4, **config)

initial_config = {
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    # Add more hyperparameters as needed
}

# 3. Mutation function
def mutate_config(config):
    new_config = copy.deepcopy(config)
    # Example: randomly tweak gamma or epsilon_decay
    if random.random() < 0.5:
        new_config["gamma"] = min(0.999, max(0.8, config["gamma"] + random.uniform(-0.05, 0.05)))
    else:
        new_config["epsilon_decay"] = min(0.999, max(0.9, config["epsilon_decay"] + random.uniform(-0.01, 0.01)))
    return new_config

# 4. Evaluation function
def evaluate_agent(agent, envs, episodes=20):
    total_reward = 0
    for env in envs:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = int(agent.select_action(state))
                next_state, reward, done = env.step(action)
                agent.store((state, action, reward, next_state, float(done)))
                agent.update(batch_size=32)
                state = next_state
                total_reward += reward
            agent.decay_epsilon()
    return total_reward / (len(envs) * episodes)

# 5. DGM loop
best_config = initial_config
best_score = float('-inf')

for generation in range(10):  # Number of self-improvement steps
    print(f"Generation {generation}")
    # Mutate
    new_config = mutate_config(best_config)
    agent = create_agent(new_config)
    # Evaluate
    score = evaluate_agent(agent, envs)
    print(f"Config: {new_config}, Score: {score}")
    # Selection
    if score > best_score:
        best_score = score
        best_config = new_config
        print("New best config found!")

print("Best config:", best_config)
