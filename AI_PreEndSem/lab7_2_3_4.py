import random
import matplotlib.pyplot as plt

# --- Binary Bandit ---
class BinaryBandit:
    def __init__(self):
        # Number of actions
        self.num_arms = 2

    # Available actions
    def get_actions(self):
        actions = []
        for i in range(self.num_arms):
            actions.append(i)
        return actions

    # First reward distribution
    def reward_dist1(self, action):
        probabilities = [0.1, 0.2]
        rand_val = random.random()
        return 1 if rand_val < probabilities[action] else 0

    # Second reward distribution
    def reward_dist2(self, action):
        probabilities = [0.8, 0.9]
        rand_val = random.random()
        return 1 if rand_val < probabilities[action] else 0

# Epsilon-greedy algorithm
def epsilon_greedy_binary(bandit, epsilon_val, iterations):
    # Initialize variables
    estimated_values = [0] * bandit.num_arms
    action_count = [0] * bandit.num_arms
    reward = 0
    total_rewards = []
    avg_rewards = [0]
    
    # Iterative approach
    for it in range(1, iterations):
        if random.random() > epsilon_val:
            selected_action = estimated_values.index(max(estimated_values))  # Greedy action
        else:
            selected_action = random.choice(bandit.get_actions())  # Explore action
        
        reward = bandit.reward_dist1(selected_action)
        total_rewards.append(reward)
        action_count[selected_action] += 1
        estimated_values[selected_action] += (reward - estimated_values[selected_action]) / action_count[selected_action]
        avg_rewards.append(avg_rewards[it-1] + (reward - avg_rewards[it-1]) / it)

    return estimated_values, avg_rewards, total_rewards

# Seed for reproducibility
random.seed(10)
bandit = BinaryBandit()
Q_values, avg_rewards_list, rewards_list = epsilon_greedy_binary(bandit, 0.2, 2000)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(avg_rewards_list)
ax1.set_title("Avg Rewards vs Iterations")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Average Reward")

ax2.plot(rewards_list)
ax2.set_title("Rewards vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Reward")

# --- Multi-Armed Bandit ---
class MultiArmBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.expected_rewards = [10] * num_arms

    # Get available actions
    def get_actions(self):
        return list(range(self.num_arms))

    # Reward calculation
    def reward(self, action):
        for i in range(self.num_arms):
            self.expected_rewards[i] += random.gauss(0, 0.1)
        return self.expected_rewards[action] + random.gauss(0, 0.01)

# Standard epsilon-greedy for multi-arm bandit
def epsilon_greedy_bandit(bandit, epsilon_val, iterations):
    estimated_values = [0] * bandit.num_arms
    action_count = [0] * bandit.num_arms
    total_rewards = []
    avg_rewards = [0]
    
    for it in range(1, iterations):
        if random.random() > epsilon_val:
            selected_action = estimated_values.index(max(estimated_values))  # Greedy
        else:
            selected_action = random.choice(bandit.get_actions())  # Explore
        
        reward = bandit.reward(selected_action)
        total_rewards.append(reward)
        action_count[selected_action] += 1
        estimated_values[selected_action] += (reward - estimated_values[selected_action]) / action_count[selected_action]
        avg_rewards.append(avg_rewards[it-1] + (reward - avg_rewards[it-1]) / it)

    return estimated_values, avg_rewards, total_rewards

random.seed(10)
multi_bandit = MultiArmBandit(10)
Q_vals, avg_rewards, rewards = epsilon_greedy_bandit(multi_bandit, 0.3, 10000)

# Printing actual and estimated rewards
print("Actual\tEstimated")
for actual, estimated in zip(multi_bandit.expected_rewards, Q_vals):
    print(f"{actual:.3f} \t {estimated:.3f}")

# Plotting results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(avg_rewards)
ax1.set_title("Avg Rewards vs Iterations")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Average Reward")

ax2.plot(rewards)
ax2.set_title("Rewards vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Reward")
fig.suptitle("Standard Epsilon-Greedy Policy")

# --- Modified epsilon-greedy ---
def epsilon_greedy_modified(bandit, epsilon_val, iterations, alpha):
    estimated_values = [0] * bandit.num_arms
    action_count = [0] * bandit.num_arms
    total_rewards = []
    avg_rewards = [0]
    
    for it in range(1, iterations):
        if random.random() > epsilon_val:
            selected_action = estimated_values.index(max(estimated_values))  # Greedy
        else:
            selected_action = random.choice(bandit.get_actions())  # Explore
        
        reward = bandit.reward(selected_action)
        total_rewards.append(reward)
        action_count[selected_action] += 1
        estimated_values[selected_action] += alpha * (reward - estimated_values[selected_action])
        avg_rewards.append(avg_rewards[it-1] + (reward - avg_rewards[it-1]) / it)

    return estimated_values, avg_rewards, total_rewards

random.seed(10)
modified_bandit = MultiArmBandit(10)
Q_vals_mod, avg_rewards_mod, rewards_mod = epsilon_greedy_modified(modified_bandit, 0.4, 10000, 0.01)

# Printing actual and estimated rewards for modified policy
print("Actual\tEstimated")
for actual, estimated in zip(modified_bandit.expected_rewards, Q_vals_mod):
    print(f"{actual:.3f} \t {estimated:.3f}")

# Plotting results for modified epsilon-greedy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(avg_rewards_mod)
ax1.set_title("Avg Rewards vs Iterations")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Average Reward")

ax2.plot(rewards_mod)
ax2.set_title("Rewards vs Iterations")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Reward")
fig.suptitle("Modified Epsilon-Greedy Policy")
