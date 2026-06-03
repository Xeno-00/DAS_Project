import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

def main():
    # Find all metrics files in /robot_metrics
    metrics_files = glob.glob("robot_metrics/agent_*_metrics.csv")
    
    if not metrics_files:
        print("No metrics files found in 'robot_metrics' directory")
        return
    
    # Concatenate all files using pandas
    all_data = []
    for file_path in metrics_files:
        agent_id = os.path.basename(file_path).split('_')[1]
        df = pd.read_csv(file_path)
        df['agent_id'] = agent_id
        all_data.append(df)
    
    combined_df = pd.concat(all_data)
    
    # Group everything by iteration - now we have num_agents rows with the same iteration
    global_metrics = combined_df.groupby('iteration').agg({
        'cost': 'sum',  # merge same iteration rows using sum
        'grad_norm': lambda x: np.sqrt(np.sum(x**2))  # merge same iteration rows using L2 norm
    }).reset_index()
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Cost
    plt.subplot(1, 2, 1)
    plt.semilogy(global_metrics['iteration'], global_metrics['cost'])
    plt.title('Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    
    # Grad_norm
    plt.subplot(1, 2, 2)
    plt.semilogy(global_metrics['iteration'], global_metrics['grad_norm'])
    plt.title('Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('global_metrics_plot.png')
    print("Global metrics plot saved to 'global_metrics_plot.png'")
    plt.show()

if __name__ == '__main__':
    main()
