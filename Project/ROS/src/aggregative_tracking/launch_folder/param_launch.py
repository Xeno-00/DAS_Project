from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import networkx as nx
import numpy as np

def compute_degrees(G):
    degrees = {}
    num_agents = G.number_of_nodes()  
    for i in range(num_agents):
        neighbors = list(G.neighbors(i))
        degrees[i] = len(neighbors) + 1  
    return degrees

def generate_launch_description():
    ld = LaunchDescription()
    np.random.seed(42) 
    
    # Parameters
    num_agents = 5
    graph_type = 'ER'  # 'cycle', 'path', 'star', 'ER'
    max_iters = 5000
    alpha = 1e-2
    beta = 0.5
    
    # Create graph
    if graph_type == 'cycle':
        G = nx.cycle_graph(num_agents)
    elif graph_type == 'path':
        G = nx.path_graph(num_agents)
    elif graph_type == 'star':
        G = nx.star_graph(num_agents - 1)
    elif graph_type == 'ER':
        G = nx.erdos_renyi_graph(num_agents, 0.5)
    
    # Compute degrees
    degrees_dict = compute_degrees(G)
    
    # Generate targets randomly
    targets = np.random.uniform(low=-5, high=5, size=(num_agents, 2))
    
    # Create agent nodes
    for i in range(num_agents):
        params = {
            'id': i,
            'neighbors': list(G.neighbors(i)),
            'target': targets[i].tolist(),
            'N': num_agents,
            'alpha': alpha,
            'beta': beta,
            'max_iters': max_iters,
            'degree': degrees_dict[i] 
        }
        
        # Add neighbors degrees
        for j in G.neighbors(i):
            params[f'degree_{j}'] = degrees_dict[j]
        
        agent_node = Node(
            package='aggregative_tracking',
            executable='robot_node',
            namespace=f'agent_{i}',
            parameters=[params],
            output='screen',
            prefix=f'xterm -title "agent_{i}" -fg white -bg black -fs 12 -hold -e',
        )
        ld.add_action(agent_node)
    
    # Create visualizer node
    visualizer_node = Node(
        package='aggregative_tracking',
        executable='visualizer_node',
        parameters=[{
            'num_agents': num_agents,
            'graph_type': graph_type
        }],
        output='screen'
    )
    ld.add_action(visualizer_node)
    
    # RViz node
    rviz_config_dir = get_package_share_directory('aggregative_tracking')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz', 'aggregative.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )
    ld.add_action(rviz_node)
    
    return ld
