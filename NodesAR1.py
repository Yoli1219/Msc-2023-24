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
N = len(nodes)  # number of nodes
T = 1000
phi = 0.9  # ar1 parameter

# Extract means and variances of nodes
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

def generate_all_ar1(means, variances, T, phi):
    latencies = np.zeros((len(means), T))  # Adjusted size based on the number of means
    for i, (mean, variance) in enumerate(zip(means, variances)):
        latencies[i, 0] = mean
        for t in range(1, T):
            latencies[i, t] = phi * latencies[i, t - 1] + np.random.normal(loc=mean * (1 - phi), scale=np.sqrt(variance))
    return latencies


all_latencies = generate_all_ar1(true_means, true_variances, T, phi)
all_latencies_means = np.mean(all_latencies, axis=1)

plt.figure(figsize=(12, 8))
for i in range(10):
    plt.plot(all_latencies[i, :], label=f"Node: {i}")
plt.xlabel('Steps', fontsize=17)
plt.ylabel('value', fontsize=17)
plt.legend()
plt.title(f'Nodes', fontsize=20)
plt.show()