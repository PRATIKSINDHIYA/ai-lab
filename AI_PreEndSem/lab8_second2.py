import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def poisson_probabilities(max_value, lambdas):
    """
    Calculate Poisson probabilities for given lambda values up to max_value.
    """
    return [poisson.pmf(np.arange(max_value + 1), lam) for lam in lambdas]

def calculate_rewards_and_transitions(state, action, rewards, costs, max_bikes, max_requests, max_returns, probabilities, discount_factor):
    """
    Compute rewards and value transitions for a given state-action pair.
    """
    loc1_bikes, loc2_bikes = state
    loc1_bikes -= action
    loc2_bikes += action

    if loc1_bikes < 0 or loc2_bikes < 0 or loc1_bikes > max_bikes or loc2_bikes > max_bikes:
        return 0, 0  # Invalid state

    transfer_cost = max(0, abs(action) - costs['free_transfer']) * costs['transfer']
    parking_cost_loc1 = costs['parking'] * (loc1_bikes > 10)
    parking_cost_loc2 = costs['parking'] * (loc2_bikes > 10)
    immediate_cost = -(transfer_cost + parking_cost_loc1 + parking_cost_loc2)

    total_reward = 0
    total_value = 0

    for requests_loc1 in range(max_requests[0] + 1):
        for requests_loc2 in range(max_requests[1] + 1):
            bikes_after_rent_loc1 = max(0, loc1_bikes - requests_loc1)
            bikes_after_rent_loc2 = max(0, loc2_bikes - requests_loc2)

            earned_rewards = (min(requests_loc1, loc1_bikes) + min(requests_loc2, loc2_bikes)) * rewards['rental']

            for returns_loc1 in range(max_returns[0] + 1):
                for returns_loc2 in range(max_returns[1] + 1):
                    next_loc1_bikes = int(min(max_bikes, bikes_after_rent_loc1 + returns_loc1))
                    next_loc2_bikes = int(min(max_bikes, bikes_after_rent_loc2 + returns_loc2))

                    prob = (
                        probabilities['requests'][0][requests_loc1] *
                        probabilities['requests'][1][requests_loc2] *
                        probabilities['returns'][0][returns_loc1] *
                        probabilities['returns'][1][returns_loc2]
                    )

                    total_reward += prob * (earned_rewards + immediate_cost)
                    total_value += prob * value_function[next_loc1_bikes, next_loc2_bikes]

    return total_reward, discount_factor * total_value

def policy_evaluation(policy, value_function, max_bikes, rewards, costs, max_requests, max_returns, probabilities, discount_factor, tolerance):
    """
    Evaluate the value function for the current policy.
    """
    delta = float('inf')
    while delta > tolerance:
        previous_value_function = value_function.copy()
        for loc1_bikes in range(max_bikes + 1):
            for loc2_bikes in range(max_bikes + 1):
                action = policy[loc1_bikes, loc2_bikes]
                reward, value = calculate_rewards_and_transitions(
                    (loc1_bikes, loc2_bikes), action, rewards, costs, max_bikes, max_requests, max_returns, probabilities, discount_factor
                )
                value_function[loc1_bikes, loc2_bikes] = reward + value
        delta = np.max(np.abs(value_function - previous_value_function))

def policy_improvement(policy, value_function, max_bikes, rewards, costs, max_requests, max_returns, probabilities, discount_factor):
    """
    Improve the policy based on the current value function.
    """
    policy_stable = True
    for loc1_bikes in range(max_bikes + 1):
        for loc2_bikes in range(max_bikes + 1):
            old_action = policy[loc1_bikes, loc2_bikes]
            best_action_value = -float('inf')
            best_action = 0

            for action in range(-costs['max_transfer'], costs['max_transfer'] + 1):
                reward, value = calculate_rewards_and_transitions(
                    (loc1_bikes, loc2_bikes), action, rewards, costs, max_bikes, max_requests, max_returns, probabilities, discount_factor
                )
                action_value = reward + value
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = action

            policy[loc1_bikes, loc2_bikes] = best_action
            if old_action != best_action:
                policy_stable = False

    return policy_stable

# Parameters
rewards = {'rental': 10}
costs = {'transfer': 2, 'parking': 4, 'free_transfer': 1, 'max_transfer': 5}
max_bikes = 20
max_requests = [12, 14]
max_returns = [12, 9]
lambdas = {'requests': [3, 4], 'returns': [3, 2]}
discount_factor = 0.9
tolerance = 0.1

# Initialization
policy = np.zeros((max_bikes + 1, max_bikes + 1), dtype=int)
value_function = np.zeros((max_bikes + 1, max_bikes + 1))

# Pre-compute probabilities
probabilities = {
    'requests': poisson_probabilities(max(max_requests), lambdas['requests']),
    'returns': poisson_probabilities(max(max_returns), lambdas['returns'])
}

# Policy iteration
policy_stable = False
iterations = 0
while not policy_stable:
    policy_evaluation(policy, value_function, max_bikes, rewards, costs, max_requests, max_returns, probabilities, discount_factor, tolerance)
    policy_stable = policy_improvement(policy, value_function, max_bikes, rewards, costs, max_requests, max_returns, probabilities, discount_factor)
    iterations += 1

# Visualization
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

c = axs[0].contour(policy, levels=np.arange(-costs['max_transfer'], costs['max_transfer'] + 1))
axs[0].set_title('Optimal Policy (Contour)')
axs[0].set_xlabel('Bikes at Location 2')
axs[0].set_ylabel('Bikes at Location 1')
fig.colorbar(c, ax=axs[0])

surf = axs[1].imshow(value_function, cmap='viridis', origin='lower', aspect='auto')
axs[1].set_title('Value Function (Surface)')
axs[1].set_xlabel('Bikes at Location 2')
axs[1].set_ylabel('Bikes at Location 1')
fig.colorbar(surf, ax=axs[1])

plt.tight_layout()
plt.show()