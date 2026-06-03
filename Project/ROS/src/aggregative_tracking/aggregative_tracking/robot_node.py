import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import os
import csv

class AggregativeRobot(Node):
    def __init__(self):
        super().__init__(
            "aggregative_robot",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        
        # Get parameters
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        self.target = np.array(self.get_parameter("target").value)
        self.N = self.get_parameter("N").value
        self.alpha = self.get_parameter("alpha").value
        self.beta = self.get_parameter("beta").value
        self.d = len(self.target)
        self.max_iters = self.get_parameter("max_iters").value
        
        # Initialize variables for aggregative tracking 
        self.z_curr = np.random.uniform(-3, 3, self.d)  # Current position
        self.s_curr = self.z_curr.copy()                # Aggregative tracker
        self.v_curr = self.beta * (self.s_curr - self.z_curr) # Gradient tracker
        self.iteration = 0
        
        # Weight matrix for neighbors (Metropolis-Hastings)
        self.weights = self.compute_weights()
        
        # Setup communication
        self.alg_publisher = self.create_publisher(
            Float64MultiArray, 
            f"/agent_{self.agent_id}/state_alg", 
            10
        )
        
        self.vis_publisher = self.create_publisher(
            Float64MultiArray, 
            f"/agent_{self.agent_id}/state_vis", 
            10
        )
        
        # Subscriptions to neighbors
        self.received_data = {}
        for j in self.neighbors:
            self.received_data[j] = {
                'z_curr': np.zeros(self.d),
                's_curr': np.zeros(self.d),
                'v_curr': np.zeros(self.d)
            }
            
            self.create_subscription(
                Float64MultiArray,
                f"/agent_{j}/state_alg",
                self.create_alg_callback(j),
                10
            )
        
        # Metrics logging
        self.log_dir = "robot_metrics"
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = open(os.path.join(self.log_dir, f"agent_{self.agent_id}_metrics.csv"), "w")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow(['iteration', 'cost', 'grad_norm'])
        
        # Timer for algorithm update
        self.update_timer = self.create_timer(0.1, self.update)
        
        # Publish initial state
        self.publish_states()
        
        self.get_logger().info(f"Agent {self.agent_id} initialized")
        self.get_logger().info(f"Target: {self.target}")
        self.get_logger().info(f"Weights: {self.weights}")

    def compute_weights(self):
        #Compute Metropolis-Hastings weights with normalization
        weights = {}
        degree_i = self.get_parameter('degree').value
        
        total_weight = 0.0
        for j in self.neighbors:
            degree_j = self.get_parameter(f"degree_{j}").value
            weights[j] = 1.0 / (1.0 + max(degree_i, degree_j))
            total_weight += weights[j]
        
        # Self weight
        weights[self.agent_id] = 1.0 - total_weight
        
        # Normalize if needed
        if abs(1.0 - sum(weights.values())) > 1e-5:
            total = sum(weights.values())
            for key in weights:
                weights[key] /= total
                
        return weights

    def create_alg_callback(self, neighbor_id):
        def callback(msg):
            try:
                data = np.array(msg.data)
                d = self.d
                self.received_data[neighbor_id]['z_curr'] = data[0:d]
                self.received_data[neighbor_id]['s_curr'] = data[d:2*d]
                self.received_data[neighbor_id]['v_curr'] = data[2*d:3*d]
            except Exception as e:
                self.get_logger().error(f"Callback error from {neighbor_id}: {str(e)}")
        return callback

    def publish_states(self):
        # Publish algorithm state
        alg_msg = Float64MultiArray()
        alg_data = np.concatenate([self.z_curr, self.s_curr, self.v_curr])
        alg_msg.data = alg_data.tolist()
        self.alg_publisher.publish(alg_msg)
        
        # Publish visualization state
        vis_msg = Float64MultiArray()
        vis_data = np.concatenate([[self.agent_id], self.z_curr, self.target])
        vis_msg.data = vis_data.tolist()
        self.vis_publisher.publish(vis_msg)

    def update(self):
        if self.iteration >= self.max_iters:
            self.get_logger().info("Max iterations reached")
            return
        
        try:
            
            # Compute gradients
            grad_local = self.z_curr - self.target
            grad_aggreg = self.beta * (self.z_curr - self.s_curr)
            grad1 = grad_local + grad_aggreg
            
            # Update position
            z_next = self.z_curr - self.alpha * (grad1 + self.v_curr)
            
            # Update aggregative tracker
            s_next = np.zeros_like(self.s_curr)
            for j in [self.agent_id] + self.neighbors:
                if j == self.agent_id:
                    s_j = self.s_curr
                else:
                    if j not in self.received_data:
                        continue
                    s_j = self.received_data[j]['s_curr']
                
                w = self.weights[j]
                s_next += w * s_j
            s_next += (z_next - self.z_curr)
            
            
            # Update gradient tracker
            grad2_curr = self.beta * (self.s_curr - self.z_curr)  
            grad2_next = self.beta * (s_next - z_next) 
            
            v_next = np.zeros(self.d)
            for j in [self.agent_id] + self.neighbors:
                if j == self.agent_id:
                    v_j = self.v_curr
                else:
                    if j not in self.received_data:
                        continue
                    v_j = self.received_data[j]['v_curr']
                
                w = self.weights[j]
                v_next += w * v_j
            v_next += (grad2_next - grad2_curr)
            
            # Compute metrics 
            cost = 0.5 * np.linalg.norm(self.z_curr - self.target)**2
            cost += 0.5 * self.beta * np.linalg.norm(self.z_curr - self.s_curr)**2
            grad1 = (self.z_curr - self.target) + self.beta * (self.z_curr - self.s_curr) +self.v_curr
            grad_norm = np.linalg.norm(grad1) 

            # Update state variables
            self.z_curr = z_next
            self.s_curr = s_next
            self.v_curr = v_next
            self.iteration += 1
            
            # Publish updated states and save metrics
            self.publish_states()
            self.metrics_writer.writerow([self.iteration, cost, grad_norm])
            self.metrics_file.flush()
            
            if self.iteration % 10 == 0:
                self.get_logger().info(f"Iteration {self.iteration}: Position={self.z_curr}")
                
        except Exception as e:
            self.get_logger().error(f"Update error: {str(e)}")
    
    def destroy_node(self):
        np.savetxt(f"robot_metrics/agent_{self.agent_id}_final_position.txt", self.z_curr)
        self.metrics_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AggregativeRobot()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()
