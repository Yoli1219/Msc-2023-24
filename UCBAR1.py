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
T = 1000  # Number of trials
alpha = 0.1  # Parameter used to calculate rewards
c_values = [0.1, 0.5]  # Different exploration factors
phi = 0.9  # AR1 process parameter

# Extract means and variances of nodes
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Get the top node with the lowest latency mean
sorted_node_indices = np.argsort(true_means)
top1_node = sorted_node_indices[0]

optimal_latencies = np.min(true_means)  # Minimum latency at each step
optimal_reward = np.exp(-alpha * optimal_latencies)


def generate_all_ar1(means, variances, T, phi):
    latencies = np.zeros((len(means), T))  # Adjusted size based on the number of means
    for i, (mean, variance) in enumerate(zip(means, variances)):
        latencies[i, 0] = mean
        for t in range(1, T):
            latencies[i, t] = phi * latencies[i, t - 1] + np.random.normal(loc=mean * (1 - phi), scale=np.sqrt(variance))
    return latencies

all_latencies = generate_all_ar1(true_means, true_variances, T, phi)  # size (10, 1000)

# UCB algorithm implementation to calculate accuracy and cumulative regret
def ucb_algorithm(num_nodes, T, alpha, c, optimal_reward):
    # Initialization
    Q = np.ones(num_nodes) * 0.1  # Initialize reward estimates for each node with a small positive number
    n = np.zeros(num_nodes)  # Number of times each node is selected
    rewards = np.zeros(T)  # Rewards for each time step
    regrets = np.zeros(T)  # Cumulative regret

    # Initialize action counts
    actions = []

    # Iterate through each round
    for t in range(1, T + 1):
        UCB = np.zeros(num_nodes)
        for i in range(num_nodes):
            if n[i] == 0:
                UCB[i] = float('inf')  # Ensure each node is selected at least once
            else:
                UCB[i] = Q[i] + c * np.sqrt(2 * np.log(t) / n[i])

        # Select the node with the highest UCB value
        action = np.argmax(UCB)
        actions.append(action)

        # Get the latency of the selected node in the current round
        action_latency = all_latencies[action, t - 1]

        # Calculate reward based on latency
        reward = np.exp(-alpha * action_latency)

        # Calculate single-step regret (ensure non-negative)
        regret = max(0, optimal_reward - reward)

        # Calculate cumulative regret
        if t == 1:
            regrets[t - 1] = regret
        else:
            regrets[t - 1] = regrets[t - 2] + regret

        # Update reward estimates
        n[action] += 1
        Q[action] = Q[action] * ((n[action] - 1) / n[action]) + reward / n[action]

        # Record the reward for the current round
        rewards[t - 1] = reward

    return Q, n, rewards, regrets, actions

# Store cumulative regrets and rewards for different c values
cumulative_regrets = {}
cumulative_rewards = {}
average_single_step_regrets_per_c = {}  # Initialize dictionary to store average single-step regret for each c value

# Run the UCB algorithm for different c values and calculate cumulative regret and rewards
for c in c_values:
    print(f"Running UCB with c = {c}")
    num_nodes = len(nodes)
    Q, n, rewards, regrets, actions = ucb_algorithm(num_nodes, T, alpha, c, optimal_reward)

    # Calculate cumulative rewards
    cumulative_reward = np.cumsum(rewards)
    cumulative_rewards[c] = cumulative_reward

    # Save cumulative regret data
    cumulative_regrets[c] = regrets

    # Calculate and store average single-step regret
    average_single_step_regret = regrets / np.arange(1, T + 1)
    average_single_step_regrets_per_c[c] = average_single_step_regret
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
    plt.plot(actions)
    plt.xlabel('Steps', fontsize=17)
    plt.ylabel('Node Selection', fontsize=17)
    plt.title(f'Node Selection for AR1 data Using UCB, c = {c}', fontsize=20)
    plt.show()

# Plot cumulative regrets for different c values on the same graph
plt.figure(figsize=(12, 8))
for c, regrets in cumulative_regrets.items():
    plt.plot(regrets, label=f'c={c}')
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Cumulative Regret', fontsize = 17)
plt.title('Cumulative Regret for AR1 data Using Different c Values', fontsize = 20)
plt.legend()
plt.show()

# Plot average single-step regret for different c values
plt.figure(figsize=(12, 8))
for c, avg_regrets in average_single_step_regrets_per_c.items():
    plt.plot(avg_regrets, label=f'c={c}')
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Average Single-Step Regret', fontsize = 17)
plt.title('Average Single-Step Regret for AR1 data Using Different c Values', fontsize = 20)
plt.legend()
plt.show()

# Plot cumulative rewards for different c values on the same graph
plt.figure(figsize=(12, 8))
for c, rewards in cumulative_rewards.items():
    plt.plot(rewards, label=f'c={c}')
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Cumulative Reward', fontsize = 17)
plt.title('Cumulative Reward for AR1 data Using Different c Values', fontsize = 20)
plt.legend()
plt.show()
