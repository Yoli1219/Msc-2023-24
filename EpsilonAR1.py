import numpy as np
import matplotlib.pyplot as plt

# Define node parameters
nodes = {
    'Node 1': {'mean': 10, 'variance': 5},
    'Node 2': {'mean': 20, 'variance': 5},
    'Node 3': {'mean': 30, 'variance': 5},
    'Node 4': {'mean': 40, 'variance': 5},
    'Node 5': {'mean': 50, 'variance': 5},
    'Node 6': {'mean': 65, 'variance': 15},
    'Node 7': {'mean': 60, 'variance': 5},
    'Node 8': {'mean': 70, 'variance': 5},
    'Node 9': {'mean': 84, 'variance': 5},
    'Node 10': {'mean': 78, 'variance': 5}
}

# Parameters
N = len(nodes)  # Number of nodes
T = 1000  # Number of trials
epsilons = [0.1, 0.6]  # List of exploration rates
alpha = 0.1  # Parameter used to calculate rewards
phi = 0.9  # AR1 generation parameter

# Extract node means and variances
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Generate AR1 latency data for all nodes
def generate_all_ar1(means, variances, T, phi):
    latencies = np.zeros((len(means), T))  # Adjusted size based on the number of means
    for i, (mean, variance) in enumerate(zip(means, variances)):
        latencies[i, 0] = mean
        for t in range(1, T):
            latencies[i, t] = phi * latencies[i, t - 1] + np.random.normal(loc=mean * (1 - phi), scale=np.sqrt(variance))
    return latencies

def calculate_cumulative_regret(regrets):
    return np.cumsum(regrets)

def calculate_cumulative_rewards(rewards):
    return np.cumsum(rewards)

all_latencies = generate_all_ar1(true_means, true_variances, T, phi)  # size (10, 1000)

all_latencies_means = np.mean(all_latencies, axis=1)  # size (10,)
optimal_latencies = np.min(true_means)  # Minimum latency at each step, size: 1
optimal_reward = np.exp(-alpha * optimal_latencies)  # size: 1

# Plot the generated latencies for each node
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.plot(all_latencies[i, :], label=f"Node: {i}")
plt.xlabel('Steps')
plt.ylabel('value')
plt.legend()
plt.title('Node Latency over Time')
plt.show()

# Record results for all epsilon values
results = {}

# Epsilon-greedy
for epsilon in epsilons:
    rewards = np.zeros(N)
    counts = np.zeros(N)
    regrets = []
    actions = []
    cumulative_rewards = 0  # Initialize cumulative rewards
    reward_actions = []

    for t in range(T):
        if np.random.rand() < epsilon:
            action = np.random.randint(N)
        else:
            action = np.argmax(rewards)

        action_latency = all_latencies[action][t]
        reward = np.exp(-alpha * action_latency)  # Calculate reward for current action
        counts[action] += 1
        n = counts[action]
        value = rewards[action]

        # Update the reward value for the current node (action)
        rewards[action] = value * ((n - 1) / n) + reward / n

        # Calculate the regret for the current step
        regret = optimal_reward - reward
        # Record the regret and action
        regrets.append(regret)
        actions.append(action)
        reward_actions.append(reward)

    cumulative_rewards = calculate_cumulative_rewards(reward_actions)
    cumulative_regret = calculate_cumulative_regret(regrets)

    optimal_nodes = np.argsort(true_means)
    counts_top1 = sum([1 for action in actions if action == optimal_nodes[0]])
    counts_top2 = sum([1 for action in actions if action in optimal_nodes[:2]])
    counts_top5 = sum([1 for action in actions if action in optimal_nodes[:5]])

    key = f"epsilon={epsilon}"
    results[key] = {
        'actions': actions,
        'regret_list': cumulative_regret,
        'regrets': regrets,  # Save single-step regret
        'cumulative_rewards': cumulative_rewards,  # Save cumulative rewards
        'top1_picks': counts_top1,
        'top2_picks': counts_top2,
        'top5_picks': counts_top5,
        'top1_accuracy': counts_top1 / T,
        'top2_accuracy': counts_top2 / T,
        'top5_accuracy': counts_top5 / T,
    }

# Print results
for epsilon, result in results.items():
    print(f"Epsilon: {epsilon}")
    print(f"Top 1 Accuracy: {result['top1_accuracy']}")
    print(f"Top 2 Accuracy: {result['top2_accuracy']}")
    print(f"Top 5 Accuracy: {result['top5_accuracy']}")
    print()

# Plot node selection for each epsilon
for epsilon, result in results.items():
    plt.figure(figsize=(12, 8))
    plt.plot(result['actions'], label=f"Epsilon: {epsilon}")
    plt.xlabel('Steps', fontsize=17)
    plt.ylabel('Node Selection', fontsize=17)
    plt.legend()
    plt.title(f'Node Selection for AR1 Data Using Epsilon = {epsilon}', fontsize=20)
    plt.show()

# Plot cumulative regret for all epsilons
plt.figure(figsize=(12, 8))
for epsilon, result in results.items():
    plt.plot(result['regret_list'], label=f"Epsilon: {epsilon}")
plt.xlabel('Steps', fontsize=17)
plt.ylabel('Cumulative Regret', fontsize=17)
plt.legend()
plt.title('Cumulative Regret for AR1 Data Using Different Epsilons', fontsize=20)
plt.show()

# Plot average single-step regret
plt.figure(figsize=(12, 8))
for epsilon, result in results.items():
    average_regrets = np.cumsum(result['regrets']) / np.arange(1, T + 1)
    plt.plot(average_regrets, label=f"Epsilon: {epsilon}")
plt.xlabel('Steps', fontsize=17)
plt.ylabel('Average Single-Step Regret', fontsize=17)
plt.title('Average Single-Step Regret Over Time for AR1 Data Using Epsilon-Greedy', fontsize=20)
plt.legend
plt.show()

# Plot cumulative rewards for each epsilon
plt.figure(figsize=(12, 8))
for epsilon, result in results.items():
    plt.plot(result['cumulative_rewards'], label=f"Epsilon: {epsilon}")
plt.xlabel('Steps', fontsize=17)
plt.ylabel('Cumulative Rewards', fontsize=17)
plt.legend()
plt.title('Cumulative Reward for AR1 Data Using Different Epsilons', fontsize=20)
plt.show()
