import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os


def main():
    # Find all metrics files in /robot_metrics
    metrics_files_cycle = glob.glob("robot_metrics/cycle/agent_*_metrics.csv")
    metrics_files_path = glob.glob("robot_metrics/path/agent_*_metrics.csv")
    metrics_files_star = glob.glob("robot_metrics/star/agent_*_metrics.csv")
    metrics_files_ER = glob.glob("robot_metrics/ER/agent_*_metrics.csv")


    # Concatenate all files using pandas
    all_data_cycle = []
    for file_path in metrics_files_cycle:
        agent_id = os.path.basename(file_path).split('_')[1]
        df = pd.read_csv(file_path)
        df['agent_id'] = agent_id
        all_data_cycle.append(df)

    combined_df_cycle = pd.concat(all_data_cycle)

    all_data_path = []
    for file_path in metrics_files_path:
        agent_id = os.path.basename(file_path).split('_')[1]
        df = pd.read_csv(file_path)
        df['agent_id'] = agent_id
        all_data_path.append(df)

    combined_df_path = pd.concat(all_data_path)

    all_data_star = []
    for file_path in metrics_files_star:
        agent_id = os.path.basename(file_path).split('_')[1]
        df = pd.read_csv(file_path)
        df['agent_id'] = agent_id
        all_data_star.append(df)

    combined_df_star = pd.concat(all_data_star)

    all_data_ER = []
    for file_path in metrics_files_ER:
        agent_id = os.path.basename(file_path).split('_')[1]
        df = pd.read_csv(file_path)
        df['agent_id'] = agent_id
        all_data_ER.append(df)

    combined_df_ER = pd.concat(all_data_ER)

    # Group everything by iteration - now we have num_agents rows with the same iteration
    global_metrics_cycle = combined_df_cycle.groupby('iteration').agg({
        'cost': 'sum',  # merge same iteration rows using sum
        'grad_norm': lambda x: np.sqrt(np.sum(x ** 2))  # merge same iteration rows using L2 norm
    }).reset_index()

    global_metrics_path = combined_df_path.groupby('iteration').agg({
        'cost': 'sum',  # merge same iteration rows using sum
        'grad_norm': lambda x: np.sqrt(np.sum(x ** 2))  # merge same iteration rows using L2 norm
    }).reset_index()

    global_metrics_star = combined_df_star.groupby('iteration').agg({
        'cost': 'sum',  # merge same iteration rows using sum
        'grad_norm': lambda x: np.sqrt(np.sum(x ** 2))  # merge same iteration rows using L2 norm
    }).reset_index()

    global_metrics_ER = combined_df_ER.groupby('iteration').agg({
        'cost': 'sum',  # merge same iteration rows using sum
        'grad_norm': lambda x: np.sqrt(np.sum(x ** 2))  # merge same iteration rows using L2 norm
    }).reset_index()


    # Plot

    # Cost
    plt.figure(figsize=(10, 6))
    plt.semilogy(global_metrics_cycle['iteration'], global_metrics_cycle['cost'],label = 'Cycle')
    plt.semilogy(global_metrics_path['iteration'], global_metrics_path['cost'],label = 'Path')
    plt.semilogy(global_metrics_star['iteration'], global_metrics_star['cost'],label = 'Star')
    plt.semilogy(global_metrics_ER['iteration'], global_metrics_ER['cost'],label = 'ER')
    plt.title('Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cost_plot.png')
    print("cost plot saved")
    plt.show()

    # Grad_norm
    plt.figure(figsize=(10, 6))
    plt.semilogy(global_metrics_cycle['iteration'], global_metrics_cycle['grad_norm'],label = 'Cycle')
    plt.semilogy(global_metrics_path['iteration'], global_metrics_path['grad_norm'],label = 'Path')
    plt.semilogy(global_metrics_star['iteration'], global_metrics_star['grad_norm'],label = 'Star')
    plt.semilogy(global_metrics_ER['iteration'], global_metrics_ER['grad_norm'],label = 'ER')
    plt.title('Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_plot.png')
    print("gradient plot saved")
    plt.show()

if __name__ == '__main__':
    main()
