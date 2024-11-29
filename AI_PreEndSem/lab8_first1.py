import numpy as np

def perform_value_iteration(grid_shape, action_directions, reward_map, transition_probs, gamma, convergence_threshold):
    """Perform value iteration to find the optimal policy."""
    value_table = np.empty(grid_shape, dtype=float)
    policy_table = np.empty(grid_shape, dtype=int)

    value_table.fill(0.0)
    delta = 0

    while True:
        delta = 0
        new_value_table = value_table.copy()

        x = 0
        while x < grid_shape[0]:
            y = 0
            while y < grid_shape[1]:
                if (x, y) in reward_map: 
                    y += 1
                    continue

                old_value = value_table[x, y]
                action_values = []

                for action, direction in action_directions.items():
                    action_value = 0
                    for probability, (dx, dy) in transition_probs[action]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
                            action_value += probability * (reward_map.get((nx, ny), -0.04) + gamma * value_table[nx, ny])
                        else:
                            action_value += probability * (reward_map.get((x, y), -0.04) + gamma * value_table[x, y])
                    action_values.append(action_value)

                new_value_table[x, y] = max(action_values)
                policy_table[x, y] = np.argmax(action_values)

                delta = max(delta, abs(old_value - new_value_table[x, y]))

                y += 1  
            x += 1 

        value_table = new_value_table

        if delta < convergence_threshold:
            break

    return value_table, policy_table

# Define the 4x3 grid and parameters
grid_shape = (4, 3)
reward_map = {
    (3, 2): 1,  
    (3, 1): -1  
}
action_directions = {
    0: (-1, 0),  
    1: (1, 0),   
    2: (0, -1),  
    3: (0, 1)   
}

# Define transition probabilities for stochastic environment
transition_probs = {
    0: [(0.8, (-1, 0)), (0.1, (0, -1)), (0.1, (0, 1))],  
    1: [(0.8, (1, 0)), (0.1, (0, -1)), (0.1, (0, 1))],  
    2: [(0.8, (0, -1)), (0.1, (-1, 0)), (0.1, (1, 0))], 
    3: [(0.8, (0, 1)), (0.1, (-1, 0)), (0.1, (1, 0))]    
}

# Set parameters
gamma = 0.9  
convergence_threshold = 1e-4

# Solve the problem for r(s) = -2 for non-terminal states
for x in range(grid_shape[0]):
    for y in range(grid_shape[1]):
        if (x, y) not in reward_map:
            reward_map[(x, y)] = -2  

value_table, policy_table = perform_value_iteration(grid_shape, action_directions, reward_map, transition_probs, gamma, convergence_threshold)

# Print results
print("Optimal Value Function:")
for x in range(grid_shape[0]):
    for y in range(grid_shape[1]):
        if (x, y) in reward_map:  
            print(f"{reward_map[(x, y)]:.2f}", end="\t")
        else:
            print(f"{value_table[x, y]:.2f}", end="\t")
    print()

print("\nOptimal Policy (0:Up, 1:Down, 2:Left, 3:Right):")
print(policy_table)