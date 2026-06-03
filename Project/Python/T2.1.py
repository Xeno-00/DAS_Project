import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42)

# Parameters
N = 5  # Number of robots
d = 2  # 2D Space
max_iters = 5000
alpha = 1e-2  # Step-size
beta = 0.5  # Tradeoff parameter

# Generate private targets in a 10x10 space
t = np.random.uniform(low=-5, high=5, size=(N, d))

# Communication graph and weight matrix
graph_types = ['cycle', 'path', 'star', 'ER']
results = {}
graphs = {}  # To save the graph for animation


# Weight matrix with Metropolis-Hastings method
def create_weight_matrix(G):
    A = nx.adjacency_matrix(G).toarray()
    W = np.zeros_like(A, dtype=float)
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                W[i, j] = 1 / (1 + max(G.degree[i], G.degree[j]))
        W[i, i] = 1 - sum(W[i])
    return W


for graph_type in graph_types:
    print(f"\nRunning {graph_type} topology...")

    # Create graph
    if graph_type == 'cycle':
        G = nx.cycle_graph(N)
    elif graph_type == 'path':
        G = nx.path_graph(N)
    elif graph_type == 'star':
        G = nx.star_graph(N - 1)
    elif graph_type == 'ER':
        G = nx.erdos_renyi_graph(N, 0.5)

    graphs[graph_type] = G
    W = create_weight_matrix(G)

    # Initialize variables
    z_curr = np.random.uniform(low=-3, high=3, size=(N, d))  # Random current positions
    s_curr = z_curr.copy()  # s_i^0 = z_i^0
    v_curr = beta * (s_curr - z_curr)

    # History storage
    positions = [z_curr.copy()]
    costs = []
    grad_norms = []
    barycenters = []
    sigma_estimates = []

    # Aggregative Tracking Algorithm
    for k in range(max_iters):
        # Update position z
        z_next = np.zeros_like(z_curr)
        for i in range(N):
            grad1 = (z_curr[i] - t[i]) + beta * (z_curr[i] - s_curr[i])
            z_next[i] = z_curr[i] - alpha * (grad1 + v_curr[i])

        # Update s_i
        s_next = np.zeros_like(s_curr)
        for i in range(N):
            for j in range(N):
                if W[i, j] > 0:  # Only neighbors
                    s_next[i] += W[i, j] * s_curr[j]
            s_next[i] += z_next[i] - z_curr[i]  # ϕ_i(z_i^{k+1}) - ϕ_i(z_i^k)

        # Update v_i
        v_next = np.zeros_like(v_curr)
        for i in range(N):
            grad2_curr = beta * (s_curr[i] - z_curr[i])
            grad2_next = beta * (s_next[i] - z_next[i])

            for j in range(N):
                if W[i, j] > 0:  # Only neighbors
                    v_next[i] += W[i, j] * v_curr[j]
            v_next[i] += grad2_next - grad2_curr

        # Update variables for next iteration
        z_curr = z_next
        s_curr = s_next
        v_curr = v_next

        positions.append(z_curr.copy())

        # Compute global cost
        cost = 0
        global_grad_norm = 0

        true_barycenter = np.mean(z_curr, axis=0) # For true barycenter vs estimate plot

        # Compute cost and gradient norm
        for i in range(N):
            cost += 0.5 * np.linalg.norm(z_curr[i] - t[i]) ** 2
            cost += 0.5 * beta * np.linalg.norm(z_curr[i] - s_curr[i]) ** 2

            grad_i = (z_curr[i] - t[i]) + beta * (z_curr[i] - s_curr[i]) + v_curr
            global_grad_norm += np.linalg.norm(grad_i) ** 2

        costs.append(cost)
        global_grad_norm = np.sqrt(global_grad_norm)
        grad_norms.append(global_grad_norm)
        barycenters.append(true_barycenter)
        sigma_estimates.append(s_curr)

    results[graph_type] = {
        'positions': np.array(positions),
        'costs': costs,
        'grad_norms': grad_norms,
        'barycenters': barycenters,
        'sigma_estimates': sigma_estimates
    }
    print(f'The true barycenter for {graph_type} is: \n{true_barycenter}')
    print(f'The estimated barycenter for {graph_type} for each agent is: \n{s_curr}')

# Plotting
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    plt.semilogy(results[graph_type]['costs'], label=graph_type)
plt.title('Total Cost Evolution')
plt.xlabel("Iteration")
plt.ylabel("Total Cost")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.savefig('Cost_evo.png')
plt.show()

plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    plt.semilogy(results[graph_type]['grad_norms'], label=graph_type)
plt.title("Global Gradient Norm")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.savefig('Glob_grad_norm.png')
plt.show()

# Barycenter error plot
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    barycenters = np.array(results[graph_type]['barycenters'])
    sigma_estimates = np.array(results[graph_type]['sigma_estimates'])

    errors = []
    for k in range(max_iters):
        true_bar = barycenters[k]  # barycenter at iteration k
        local_errors = [np.linalg.norm(sigma_estimates[k][agent] - true_bar) for agent in range(N)] # mean error between local estimates and true barycenter
        errors.append(np.mean(local_errors))

    plt.semilogy(errors, label=graph_type)
plt.title("Barycenter error")
plt.xlabel("Iteration")
plt.ylabel("Mean Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.savefig('Barycenter_error.png')
plt.show()

# Consensus error
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    sigma_estimates = np.array(results[graph_type]['sigma_estimates'])

    consensus_errors = []
    for k in range(len(sigma_estimates)):
        estimates_k = sigma_estimates[k]
        mean_estimate = np.mean(estimates_k, axis=0)
        consensus_error = np.mean([np.linalg.norm(estimates_k[agent] - mean_estimate)
                                    for agent in range(N)])
        consensus_errors.append(consensus_error)

    plt.semilogy(consensus_errors)
plt.title("Consensus Error")
plt.xlabel("Iteration")
plt.ylabel("Consensus Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.savefig('Consensus_error.png')
plt.show()

# Animation for desired topology
fig, ax = plt.subplots(figsize=(10, 8))
graph_type = 'star'
pos_history = results[graph_type]['positions']
G = graphs[graph_type]
Adj = nx.adjacency_matrix(G).toarray()

# Show 1 frame every 5 iterations
step = 5
frames_to_show = list(range(0, max_iters, step))


def update(frame_idx):
    frame = frame_idx * step
    if frame >= len(pos_history):
        frame = len(pos_history) - 1

    ax.clear()
    z = pos_history[frame]
    barycenter = np.mean(z, axis=0)

    # Plot elements
    ax.scatter(z[:, 0], z[:, 1], c='blue', s=80, label='Robots')
    ax.scatter(t[:, 0], t[:, 1], c='red', marker='x', s=100, label='Targets')
    ax.scatter(barycenter[0], barycenter[1], c='green', marker='*', s=200, label='Barycenter')

    # Connections
    for i in range(N):
        ax.plot([z[i, 0], t[i, 0]], [z[i, 1], t[i, 1]], 'k--', alpha=0.3)
        ax.plot([z[i, 0], barycenter[0]], [z[i, 1], barycenter[1]], 'g-', alpha=0.2)

    # Communication graph
    for i in range(N):
        for j in range(i + 1, N):
            if Adj[i, j] > 0:
                ax.plot([z[i, 0], z[j, 0]], [z[i, 1], z[j, 1]],
                        color='#5286c6', linewidth=1.5, alpha=0.7,
                        label='Graph edges' if i == 0 and j == 1 else "")

    ax.set_title(f"Aggregative Optimization: Iteration {frame}/{max_iters}")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.legend()
    ax.grid(True)
    return ax


# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(frames_to_show),
    interval=30,
    blit=False
)

plt.show()