import numpy as np
import matplotlib.pyplot as plt

# Defining node parameters
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
N = len(nodes)  # Number of nodes
T = 1000  # Number of trials
alpha = 0.1 # Parameter used in calculating reward
epsilon = 0.9

# Extract means and variances of the nodes
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Generate latency data for all nodes
def generate_all_iid(means, T):
    all_latencies = []
    for mean in means:
        latencies = np.random.normal(loc=mean, scale=np.sqrt(5), size=int(T))  # Ensure T is an integer
        all_latencies.append(latencies)
    return np.array(all_latencies)

all_latencies = generate_all_iid(true_means, T)
optimal_latencies = np.min(true_means)  # Minimum latency at each step
optimal_reward = np.exp(-alpha * optimal_latencies)
print(optimal_reward, '===')

def calculate_cumulative_regret(regrets):
    return np.cumsum(regrets)


all_latencies = generate_all_iid(true_means, T)
optimal_latencies = np.min(all_latencies, axis=0)  # Minimum latency at each step

# Adaptive epsilon-greedy
rewards = np.zeros(N)
counts = np.zeros(N)
regrets = []
actions = []
cumulative_rewards = []  # Store cumulative rewards
cumulative_reward = 0  # Initialize cumulative reward

for t in range(1, T + 1):  # Start t from 1 to avoid division by zero
    # Adjust epsilon over time
    epsilon_t = max(0.01, t ** (-1 / 3))
    if np.random.rand() < epsilon_t:
        action = np.random.randint(int(N))  # Ensure N is an integer
    else:
        action = np.argmax(rewards)

    action_latency = all_latencies[action][t-1]  # t-1, because index starts at 0
    reward = np.exp(-alpha * action_latency)
    counts[action] += 1
    n = counts[action]
    value = rewards[action]

    rewards[action] = value * ((n - 1) / n) + reward / n

    # Update cumulative reward
    cumulative_reward += reward
    cumulative_rewards.append(cumulative_reward)

    regret = optimal_reward - reward

    regrets.append(regret)
    actions.append(action)

cumulative_regret = calculate_cumulative_regret(regrets)

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
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Node Selection', fontsize = 17)
plt.title('Node Selection for IID data Using Adaptive', fontsize = 20)
plt.show()

# Plot cumulative regret over time
plt.figure(figsize=(12, 8))
plt.plot(cumulative_regret)
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Cumulative Regret', fontsize = 17)
plt.title('Cumulative Regret for IID data Using Adaptive', fontsize = 20)
plt.show()


# Plot average single-step regret over time
plt.figure(figsize=(12, 8))
average_regrets = np.cumsum(regrets) / np.arange(1, T+1)
plt.plot(average_regrets)
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Average Single-Step Regret', fontsize = 17)
plt.title('Average Single-Step Regret for IID data Using Adaptive', fontsize = 20)
plt.show()

# Plot cumulative rewards over time
plt.figure(figsize=(12, 8))
plt.plot(cumulative_rewards)
plt.xlabel('Steps', fontsize = 17)
plt.ylabel('Cumulative Reward', fontsize = 17)
plt.title('Cumulative Reward for IID data Using Adaptive', fontsize = 20)
plt.show()