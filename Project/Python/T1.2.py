import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
N = 10  # Number of robots
d = 2  # 2D Space
NT = 3  # Number of targets
max_iters = 10000
alpha = 1e-3

# Robot positions
p = np.random.uniform(low=-3, high=3, size=(N, d))

# True target positions
z_true = np.random.uniform(low=-3, high=3, size=(NT, d))

# Simulated noisy distances
dists = np.zeros((N, NT))
noise_std = 0.1
for i in range(N):
    for tau in range(NT):
        dists[i, tau] = np.linalg.norm(p[i] - z_true[tau]) + np.random.normal(0, noise_std)


# Local cost function
def local_cost(z_actual, i):
    cost = 0
    grad = np.zeros_like(z_actual)
    for tau in range(NT):
        z_tau = z_actual[tau * d: (tau + 1) * d]
        pi = p[i]
        diff = np.linalg.norm(z_tau - pi) ** 2
        error = dists[i, tau] ** 2 - diff
        cost += error ** 2

        grad_tau = -4 * error * (z_tau - pi)
        grad[tau * d: (tau + 1) * d] = grad_tau
    return cost, grad

# Weight matrix with Metropolis-Hastings method
def create_weight_matrix(G):
    A = nx.adjacency_matrix(G).toarray()
    W = np.zeros_like(A, dtype=float)
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                W[i, j] = 1 / (1 + max(G.degree[i], G.degree[j]))
        W[i, i] = 1 - sum(W[i])
    return A, W

graph_types = ['cycle'
    , 'path'
    , 'star'
    , 'ER'
               ]
results = {}

for graph_type in graph_types:
    print(f"\nRunning {graph_type} topology...")

    if graph_type == 'cycle':
        G = nx.cycle_graph(N)
    elif graph_type == 'path':
        G = nx.path_graph(N)
    elif graph_type == 'star':
        G = nx.star_graph(N - 1)
    elif graph_type == 'ER':
        G = nx.erdos_renyi_graph(N, 0.5)

    Adj, W = create_weight_matrix(G)

    # Initialization
    dim_z = d * NT
    z = np.ones((max_iters, N, dim_z))
    s = np.zeros((max_iters, N, dim_z))
    grad_new = np.zeros((N, d))
    grad_old = np.zeros((N, d))

    # Initialize near robots
    for i in range(N):
        for tau in range(NT):
            offset = np.random.uniform(-0.5, 0.5, size=d)
            z[0, i, tau * d:(tau + 1) * d] = p[i] + offset

    # Initialize gradients
    for i in range(N):
        _, s[0, i] = local_cost(z[0, i], i)

    # lists for metrics
    cost_history = []
    gradnorm_history = []
    target_error_history = []

    for k in range(max_iters - 1):
        total_cost = 0
        grad_sum = np.zeros(dim_z)

        # Update positions
        z[k + 1] = W @ z[k] - alpha * s[k]

        # Update gradient tracking
        s[k + 1] = W @ s[k]
        for i in range(N):
            _, grad_new = local_cost(z[k + 1, i], i)
            _, grad_old = local_cost(z[k, i], i)
            s[k + 1, i] += grad_new - grad_old

        # Calculate metrics on current positions (z[k+1])
        for i in range(N):
            cost_val, grad_val = local_cost(z[k + 1, i], i)
            total_cost += cost_val
            grad_sum += grad_val

        # Calculate target error
        current_error = 0
        for i in range(N):
            z_estimated = z[k + 1, i].reshape(NT, d)
            current_error += np.mean(np.linalg.norm(z_estimated - z_true, axis=1))
        target_error = current_error / N

        cost_history.append(total_cost)
        gradnorm_history.append(np.linalg.norm(grad_sum))
        target_error_history.append(target_error)

    results[graph_type] = {
        'cost_history': cost_history,
        'gradnorm_history': gradnorm_history,
        'target_error': target_error_history
    }
    print(f"\nFinal Target Estimation Error {graph_type}:\n\t{results[graph_type]['target_error'][-1]}")

# Plotting
# Cost Function Evolution
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    plt.semilogy(results[graph_type]['cost_history'], label=graph_type)
plt.title("Cost Function Evolution")
plt.xlabel("Iteration")
plt.ylabel("Total Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Total_cost_evo.png')
plt.show()

# Global Gradient Norm
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    plt.semilogy(results[graph_type]['gradnorm_history'], label=graph_type)
plt.title("Global Gradient Norm Evolution")
plt.xlabel("Iteration")
plt.ylabel("||∑ grad l_i||")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Global_gradient_norm.png')
plt.show()

# Target Estimation Error
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    plt.semilogy(results[graph_type]['target_error'], label=graph_type)
plt.title("Target Estimation Error")
plt.xlabel("Iteration")
plt.ylabel("Average Position Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Target_est_err.png')
plt.show()

# True vs Estimated Positions
plt.figure(figsize=(10, 8))
plt.scatter(p[:, 0], p[:, 1], c='red', s=80, label='Robots')
plt.scatter(z_true[:, 0], z_true[:, 1], c='black', marker='x', s=100, label='True Targets')

# Get the final consensus estimate
final_estimates = np.zeros((N, NT, d))
for i in range(N):
    final_estimates[i] = z[-1, i].reshape(NT, d)
consensus_estimate = np.mean(final_estimates, axis=0)

plt.scatter(consensus_estimate[:, 0], consensus_estimate[:, 1],
            c='green', marker='*', s=150, label='Consensus Estimate')

# Connect estimates to true positions with a dashed line
for tau in range(NT):
    plt.plot([z_true[tau, 0], consensus_estimate[tau, 0]],
             [z_true[tau, 1], consensus_estimate[tau, 1]],
             'k--', alpha=0.3)
graph_type = graph_types[-1]
plt.legend()
plt.title(f"True vs Estimated: {graph_type}")
plt.grid(True)
plt.tight_layout()
#plt.savefig(f'TargetsVSEstimated_{graph_type}.png')
plt.show()