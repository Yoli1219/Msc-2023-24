import numpy as np
import matplotlib.pyplot as plt

# Define node parameters
nodes = {
    'Node 1': {'mean': 10, 'variance': 5},
    'Node 2': {'mean': 20, 'variance': 5},
    'Node 3': {'mean': 30, 'variance': 5},
    'Node 4': {'mean': 40, 'variance': 5},
    'Node 5': {'mean': 50, 'variance': 5},
    'Node 6': {'mean': 55, 'variance': 15},
    'Node 7': {'mean': 60, 'variance': 5},
    'Node 8': {'mean': 70, 'variance': 5},
    'Node 9': {'mean': 84, 'variance': 5},
    'Node 10': {'mean': 78, 'variance': 5}
}

# Parameters
N = len(nodes)  # Number of nodes (arms)
T = 1000  # Number of trials (rounds)
alpha = 0.1  # Parameter for calculating delay reward
m = 0.5  # Weight parameter to balance delay reward and load reward
phi = 0.9  # AR(1) process parameter

# Extract node means and variances
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Generate latency and load data
def generate_latency_ar1(means, variances, T, phi):
    latencies = np.zeros((len(means), T))  # Adjusted size based on the number of means
    for i, (mean, variance) in enumerate(zip(means, variances)):
        latencies[i, 0] = mean
        for t in range(1, T):
            latencies[i, t] = phi * latencies[i, t - 1] + np.random.normal(loc=mean * (1 - phi), scale=np.sqrt(variance))
    return latencies

# Generate latency and load data
def generate_load_ar1(means, variances, T, phi):
    loads = np.zeros((len(means), T))  # Adjusted size based on the number of means
    for i, (mean, variance) in enumerate(zip(means, variances)):
        loads[i, 0] = mean
        for t in range(1, T):
            loads[i, t] = phi * loads[i, t - 1] + np.random.normal(loc=mean * (1 - phi), scale=np.sqrt(variance))
    return loads

# Generate data
all_latencies = generate_latency_ar1(true_means, true_variances, T, phi)
all_loads = generate_load_ar1(true_means, true_variances, T, phi)

# Calculate reward function
def calculate_rewards_latency(action_latency, alpha):
    reward_latency = np.exp(-alpha * action_latency)
    return reward_latency

def calculate_rewards_load(action_load):
    reward_load = 1 / (1 + action_load)
    return reward_load

# Get combined reward
def get_reward(action_latency, action_load, alpha, m):
    reward_latency = calculate_rewards_latency(action_latency, alpha)
    reward_load = calculate_rewards_load(action_load)
    reward = m * reward_latency + (1 - m) * reward_load
    return reward

# Calculate expected rewards and find the optimal node
combine_rewards = []
for i in range(N):
    mean_latency = true_means[i]
    mean_load = true_means[i]

    reward_latency = np.exp(-alpha * mean_latency)
    reward_load = 1 / (1 + mean_load)
    expected_reward = m * reward_latency + (1 - m) * reward_load

    combine_rewards.append(expected_reward)

# Run the EXP3 algorithm and record the counts of selections for the top 1, top 2, and top 5 nodes
gamma_values = [0.2, 0.8]  # Adjust gamma values for more exploration
cumulative_rewards_per_gamma = {}  # Store cumulative rewards for each gamma
average_single_step_regrets_per_gamma = {}  # Store average single-step regrets for each gamma
cumulative_regrets_per_gamma = {}  # Store cumulative regrets for each gamma

# Get the expected reward of the best node (maximum)
best_expected_reward = max(combine_rewards)

for gamma in gamma_values:
    weights = np.ones(N)
    actions = np.zeros(T, dtype=int)  # Initialize actions array
    regrets = np.zeros(T)  # Initialize regret array
    cumulative_rewards = np.zeros(T)  # Initialize cumulative reward array
    cumulative_reward = 0  # Initialize cumulative reward
    cumulative_regrets = np.zeros(T)  # Initialize cumulative regret array

    # Simulate the EXP3 process
    for t in range(T):
        # Calculate total weight Wt
        total_weight = np.sum(weights)

        # Calculate selection probabilities
        probabilities = (1 - gamma) * (weights / total_weight) + (gamma / N)

        # Select a node based on the selection probabilities
        action = np.random.choice(N, p=probabilities)
        actions[t] = action  # Record the currently selected node

        # Get the latency and load values
        action_latency = all_latencies[action, t]
        action_load = all_loads[action, t]

        # Calculate the current actual reward
        reward = get_reward(action_latency, action_load, alpha, m)

        # Calculate the estimated reward
        estimated_reward = reward / probabilities[action]

        # Update weights
        weights[action] *= np.exp(gamma * estimated_reward / N)

        # Cumulative reward
        cumulative_reward += reward
        cumulative_rewards[t] = cumulative_reward

        # Calculate single-step regret (best expected reward - actual reward)
        regret = best_expected_reward - reward
        regrets[t] = regret

    cumulative_rewards_per_gamma[gamma] = cumulative_rewards  # Store cumulative rewards for the current gamma
    cumulative_regrets_per_gamma[gamma] = np.cumsum(regrets)  # Store cumulative regrets for the current gamma

    # Calculate and store average single-step regret
    average_single_step_regret = np.cumsum(regrets) / np.arange(1, T + 1)
    average_single_step_regrets_per_gamma[gamma] = average_single_step_regret

    optimal_nodes = np.argsort(true_means)
    counts_top1 = sum([1 for action in actions if action == optimal_nodes[0]])
    counts_top2 = sum([1 for action in actions if action in optimal_nodes[:2]])
    counts_top5 = sum([1 for action in actions if action in optimal_nodes[:5]])

    # Print results
    print(f"Top 1 Accuracy: {counts_top1 / T}")
    print(f"Top 2 Accuracy: {counts_top2 / T}")
    print(f"Top 5 Accuracy: {counts_top5 / T}")

    # Plot node selection over time
    plt.figure(figsize=(12, 8))
    plt.plot(actions, label=f'Gamma = {gamma}')
    plt.xlabel('Steps', fontsize=17)
    plt.ylabel('Node Selections', fontsize=17)
    plt.title(f'Node Selection AR1 Using EXP3, Gamma = {gamma}', fontsize=20)
    plt.legend()
    plt.show()

# Plot cumulative regret for different gamma values
plt.figure(figsize=(12, 8))
for gamma in gamma_values:
    cumulative_regret = np.cumsum(average_single_step_regrets_per_gamma[gamma])  # Cumulative regret for the corresponding gamma
    plt.plot(cumulative_regret, label=f'Gamma = {gamma}')
plt.xlabel('Time Steps', fontsize=17)
plt.ylabel('Cumulative Regret', fontsize=17)
plt.title('Cumulative Regret for AR1 data Using Different Gamma', fontsize=20)
plt.legend()
plt.show()

# Plot average single-step regret for different gamma values
plt.figure(figsize=(12, 8))
for gamma in gamma_values:
    plt.plot(average_single_step_regrets_per_gamma[gamma], label=f'Gamma = {gamma}')
plt.xlabel('Time Steps', fontsize=17)
plt.ylabel('Average Single-Step Regret', fontsize=17)
plt.title('Average Single-Step Regret Over Time for AR1 data Using Different Gamma', fontsize=20)
plt.legend()
plt.show()

# Plot cumulative rewards for different gamma values
plt.figure(figsize=(12, 8))
for gamma in gamma_values:
    plt.plot(cumulative_rewards_per_gamma[gamma], label=f'Gamma = {gamma}')
plt.xlabel('Time Steps', fontsize = 17)
plt.ylabel('Cumulative Reward', fontsize = 17)
plt.title('Cumulative Reward Over Time for AR1 data Using EXP3', fontsize = 20)
plt.legend()
plt.show()
