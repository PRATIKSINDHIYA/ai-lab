import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters
expected_rentals = [3, 4]  # Expected rental requests (location 1 and 2)
expected_returns = [3, 2]  # Expected returns (location 1 and 2)
rental_reward = 10  # Reward per rental
transfer_cost_per_bike = 2  # Cost per bike transferred (after the first bike)
parking_penalty = 4  # Penalty for exceeding 10 bikes at a location
free_transfer_limit = 1  # Free bike transfer limit from location 1 to 2
discount_factor = 0.9  # Discount factor
max_bike_storage = 20  # Maximum bikes at each location
max_transfer_limit = 5  # Maximum bikes transferred overnight

# Initialization
policy_matrix = np.zeros((max_bike_storage + 1, max_bike_storage + 1), dtype=int)  # Initial policy
value_matrix = np.zeros_like(policy_matrix, dtype=float)  # Value function initialization
is_policy_stable = False
iteration_counter = 0

def poisson_probabilities(max_range, lamda):
    """Compute Poisson probabilities."""
    return poisson.pmf(np.arange(max_range + 1), lamda)

# Precompute probabilities
rental_probs_loc1 = poisson_probabilities(12, expected_rentals[0])
rental_probs_loc2 = poisson_probabilities(14, expected_rentals[1])
return_probs_loc1 = poisson_probabilities(12, expected_returns[0])
return_probs_loc2 = poisson_probabilities(9, expected_returns[1])

while not is_policy_stable:
    # Policy Evaluation
    delta = float('inf')
    convergence_threshold = 0.1  # Convergence tolerance

    while delta > convergence_threshold:
        old_value_matrix = value_matrix.copy()
        for s1 in range(max_bike_storage + 1):
            for s2 in range(max_bike_storage + 1):
                action = policy_matrix[s1, s2]
                s1_post_action = s1 - action
                s2_post_action = s2 + action

                # Validity checks
                if s1_post_action < 0 or s2_post_action < 0 or s1_post_action > max_bike_storage or s2_post_action > max_bike_storage:
                    continue

                # Costs
                transfer_cost = max(0, abs(action) - free_transfer_limit) * transfer_cost_per_bike
                parking_cost = (s1_post_action > 10) * parking_penalty + (s2_post_action > 10) * parking_penalty
                immediate_cost = -(transfer_cost + parking_cost)

                # Value update
                expected_value = 0
                for n1 in range(13):  # Rental requests location 1
                    for n2 in range(15):  # Rental requests location 2
                        s1_rentals = max(0, s1_post_action - n1)
                        s2_rentals = max(0, s2_post_action - n2)

                        for n3 in range(13):  # Returns location 1
                            for n4 in range(10):  # Returns location 2
                                s1_returns = min(s1_rentals + n3, max_bike_storage)
                                s2_returns = min(s2_rentals + n4, max_bike_storage)

                                probability = (
                                    rental_probs_loc1[n1]
                                    * rental_probs_loc2[n2]
                                    * return_probs_loc1[n3]
                                    * return_probs_loc2[n4]
                                )
                                reward = (min(n1, s1_post_action) + min(n2, s2_post_action)) * rental_reward
                                expected_value += probability * (reward + discount_factor * old_value_matrix[s1_returns, s2_returns])

                value_matrix[s1, s2] = immediate_cost + expected_value

        delta = np.max(np.abs(value_matrix - old_value_matrix))

    # Policy Improvement
    is_policy_stable = True
    for s1 in range(max_bike_storage + 1):
        for s2 in range(max_bike_storage + 1):
            old_action = policy_matrix[s1, s2]
            action_values = []

            for action in range(-max_transfer_limit, max_transfer_limit + 1):
                s1_post_action = s1 - action
                s2_post_action = s2 + action

                if s1_post_action < 0 or s2_post_action < 0 or s1_post_action > max_bike_storage or s2_post_action > max_bike_storage:
                    continue

                transfer_cost = max(0, abs(action) - free_transfer_limit) * transfer_cost_per_bike
                parking_cost = (s1_post_action > 10) * parking_penalty + (s2_post_action > 10) * parking_penalty
                immediate_cost = -(transfer_cost + parking_cost)

                expected_value = 0
                for n1 in range(13):
                    for n2 in range(15):
                        s1_rentals = max(0, s1_post_action - n1)
                        s2_rentals = max(0, s2_post_action - n2)
                        for n3 in range(13):
                            for n4 in range(10):
                                s1_returns = min(s1_rentals + n3, max_bike_storage)
                                s2_returns = min(s2_rentals + n4, max_bike_storage)

                                probability = (
                                    rental_probs_loc1[n1]
                                    * rental_probs_loc2[n2]
                                    * return_probs_loc1[n3]
                                    * return_probs_loc2[n4]
                                )
                                reward = (min(n1, s1_post_action) + min(n2, s2_post_action)) * rental_reward
                                expected_value += probability * (reward + discount_factor * value_matrix[s1_returns, s2_returns])

                action_values.append(immediate_cost + expected_value)

            optimal_action = range(-max_transfer_limit, max_transfer_limit + 1)[np.argmax(action_values)]
            policy_matrix[s1, s2] = optimal_action

            if old_action != optimal_action:
                is_policy_stable = False

    iteration_counter += 1

# Visualization
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Optimal policy plot
c = axs[0].contour(policy_matrix, levels=np.arange(-max_transfer_limit, max_transfer_limit + 1))
axs[0].set_title('Optimal Policy (Contour)')
axs[0].set_xlabel('Bikes at Location 2')
axs[0].set_ylabel('Bikes at Location 1')
fig.colorbar(c, ax=axs[0])

# Value function plot
surf = axs[1].imshow(value_matrix, cmap='viridis', origin='lower', aspect='auto')
axs[1].set_title('Value Function (Surface)')
axs[1].set_xlabel('Bikes at Location 2')
axs[1].set_ylabel('Bikes at Location 1')
fig.colorbar(surf, ax=axs[1])

plt.tight_layout()
plt.show()