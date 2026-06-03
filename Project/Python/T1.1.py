import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
N = 10 # Number of agents
d = 5
max_iters = 10000
alpha = 1e-2 # Step-size

# Cost functions: Q positive definite, r random
Q = [np.diag(np.random.rand(d)) for _ in range(N)]
r = [np.random.randn(d) for _ in range(N)]

#Q and r test
#Q = [np.eye(d) for _ in range(N)]
#r = [np.zeros(d) for _ in range(N)]

def cost_function(z, Q, r):
    cost = 0.5 * z.T @ Q @ z + r.T @ z
    grad = Q @ z + r
    return cost, grad


# Optimal solution
Q_sum = sum(Q)
r_sum = sum(r)
z_opt = -np.linalg.solve(Q_sum, r_sum)
J_opt, grad_opt = cost_function(z_opt, Q_sum, r_sum)


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

    # Initialize variables
    z = np.ones((max_iters, N, d))
    s = np.zeros((max_iters, N, d))

    # Initialize s with gradient
    for i in range(N):
        _, s[0, i] = cost_function(z[0, i], Q[i], r[i])

    cost_history = []
    gradnorm_history = []
    z_avg_history = np.zeros((max_iters, d))

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
        z_avg = np.mean(z[k+1], axis=0)
        z_avg_history[k] = z_avg

        cost_history.append(total_cost)
        gradnorm_history.append(np.linalg.norm(grad_sum))

    results[graph_type] = {
        'cost_history': cost_history,
        'gradnorm_history': gradnorm_history,
        'z_avg_history': z_avg_history
    }

# Plotting
plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    cost_vals = results[graph_type]['cost_history']
    plt.plot(np.array(cost_vals), label=graph_type)
plt.title("Total Cost Evolution")
plt.xlabel("Iteration")
plt.ylabel("J(z)")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Total_cost_evo.png')
plt.show()

plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    cost_vals = results[graph_type]['cost_history']
    plt.semilogy(np.abs(np.array(cost_vals) - J_opt), label=graph_type)
plt.title("Total cost error")
plt.xlabel("Iteration")
plt.ylabel("|J - J*|")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Total_cost_error.png')
plt.show()

plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    plt.semilogy(results[graph_type]['gradnorm_history'], label=graph_type)
plt.title("Global Gradient Norm")
plt.xlabel("Iteration")
plt.ylabel("||∑ grad l_i||")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Global_gradient_norm.png')
plt.show()

plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    z_avg_hist = results[graph_type]['z_avg_history']
    consensus_error = [np.linalg.norm(z[k] - z_avg_hist[k], 'fro') / N for k in range(max_iters - 1)]
    plt.semilogy(consensus_error, label=graph_type)
plt.title("Consensus Error")
plt.xlabel("Iteration")
plt.ylabel("||z_i - z_avg||")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Consensus_error.png')
plt.show()

plt.figure(figsize=(10, 6))
for graph_type in graph_types:
    z_avg_hist = results[graph_type]['z_avg_history']
    opt_distance = [np.linalg.norm(z_avg_hist[k] - z_opt) for k in range(max_iters - 1)]
    plt.semilogy(opt_distance, label=graph_type)
plt.title("Distance from Optimal Solution")
plt.xlabel("Iteration")
plt.ylabel("||z_avg - z*||")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('Distance_opt_sol.png')
plt.show()