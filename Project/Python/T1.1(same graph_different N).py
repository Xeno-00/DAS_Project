import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
N_a = [3, 10, 20]  # Different numbers of agents
d = 5
max_iters = 20000
alpha = 1e-2
graph_type = 'cycle'  # Fixed graph topology


def cost_function(z, Q, r):
    cost = 0.5 * z.T @ Q @ z + r.T @ z
    grad = Q @ z + r
    return cost, grad


# Weight matrix with Metropolis-Hastings method
def create_weight_matrix(G, N):
    A = nx.adjacency_matrix(G).toarray()
    W = np.zeros_like(A, dtype=float)
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                W[i, j] = 1 / (1 + max(G.degree[i], G.degree[j]))
        W[i, i] = 1 - sum(W[i])
    return A, W


results = {}

for N in N_a:  # Loop over different agent counts
    print(f"\nRunning {graph_type} topology with {N} agents...")

    # Create graph based on fixed type
    if graph_type == 'cycle':
        G = nx.cycle_graph(N)
    elif graph_type == 'path':
        G = nx.path_graph(N)
    elif graph_type == 'star':
        G = nx.star_graph(N - 1)
    elif graph_type == 'ER':
        G = nx.erdos_renyi_graph(N, 0.5)

    Adj, W = create_weight_matrix(G, N)

    # Initialize variables
    z = np.ones((max_iters, N, d))
    s = np.zeros((max_iters, N, d))

    # Cost functions: Q positive definite, r random
    Q = [np.diag(np.random.rand(d)) for _ in range(N)]
    r = [np.random.randn(d) for _ in range(N)]

    # Optimal solution
    Q_sum = sum(Q)
    r_sum = sum(r)
    z_opt = -np.linalg.solve(Q_sum, r_sum)
    J_opt, grad_opt = cost_function(z_opt, Q_sum, r_sum)

    # Initialize s with gradient
    for i in range(N):
        _, s[0, i] = cost_function(z[0, i], Q[i], r[i])

    cost_history = []
    gradnorm_history = []
    consensus_error_history = []
    opt_distance_history = []

    for k in range(max_iters - 1):
        total_cost = 0
        grad_sum = np.zeros(d)

        # Compute cost and gradient before update
        for i in range(N):
            cost_val, grad_val = cost_function(z[k, i], Q[i], r[i])
            total_cost += cost_val
            grad_sum += grad_val

        # Update states
        z[k + 1] = W @ z[k] - alpha * s[k]

        # Update tracking variables
        grad_new = np.zeros((N, d))
        grad_old = np.zeros((N, d))
        s[k + 1] = W @ s[k]
        for i in range(N):
            _, grad_new[i] = cost_function(z[k + 1, i], Q[i], r[i])
            _, grad_old[i] = cost_function(z[k, i], Q[i], r[i])

        s[k + 1] += grad_new - grad_old

        # Compute metrics
        z_avg = np.mean(z[k + 1], axis=0)

        # Consensus error
        consensus_error = np.mean([np.linalg.norm(z_i - z_avg) for z_i in z[k + 1]])
        consensus_error_history.append(consensus_error)

        # Distance from the optimal solution
        opt_distance = np.linalg.norm(z_avg - z_opt)
        opt_distance_history.append(opt_distance)

        cost_history.append(total_cost)
        gradnorm_history.append(np.linalg.norm(grad_sum))

    # Store results for this N
    results[N] = {
        'cost_history': cost_history,
        'gradnorm_history': gradnorm_history,
        'consensus_error': consensus_error_history,
        'opt_distance': opt_distance_history,
        'J_opt': J_opt
    }

# Plotting
plt.figure(figsize=(10, 6))
for N in N_a:
    plt.plot(results[N]['cost_history'], label=f'N={N}')
plt.title(f"Total Cost Evolution ({graph_type} topology)")
plt.xlabel("Iteration")
plt.ylabel("J(z)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Total_cost_evo.png')
plt.show()

plt.figure(figsize=(10, 6))
for N in N_a:
    cost_vals = results[N]['cost_history']
    J_opt = results[N]['J_opt']
    plt.semilogy(np.abs(np.array(cost_vals) - J_opt), label=f'N={N}')
plt.title(f"Total Cost Error ({graph_type} topology)")
plt.xlabel("Iteration")
plt.ylabel("|J - J*|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Total_cost_error.png')
plt.show()

plt.figure(figsize=(10, 6))
for N in N_a:
    plt.semilogy(results[N]['gradnorm_history'], label=f'N={N}')
plt.title(f"Global Gradient Norm ({graph_type} topology)")
plt.xlabel("Iteration")
plt.ylabel("||∑ grad l_i||")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Global_gradient_norm.png')
plt.show()

plt.figure(figsize=(10, 6))
for N in N_a:
    plt.semilogy(results[N]['consensus_error'], label=f'N={N}')
plt.title(f"Consensus Error ({graph_type} topology)")
plt.xlabel("Iteration")
plt.ylabel("Average ||z_i - z_avg||")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Consensus_error.png')
plt.show()

plt.figure(figsize=(10, 6))
for N in N_a:
    plt.semilogy(results[N]['opt_distance'], label=f'N={N}')
plt.title(f"Distance from Optimal Solution ({graph_type} topology)")
plt.xlabel("Iteration")
plt.ylabel("||z_avg - z*||")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Distance_opt_sol.png')
plt.show()




