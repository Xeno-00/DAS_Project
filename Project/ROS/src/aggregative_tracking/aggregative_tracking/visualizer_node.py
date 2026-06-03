import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import numpy as np
import time

class AggregativeVisualizer(Node):
    def __init__(self):
        super().__init__('aggregative_visualizer')
        
        # Parameters
        self.declare_parameter('num_agents', 5)
        self.num_agents = self.get_parameter('num_agents').value
        
        # Agent states storage
        self.agent_states = {}
        self.last_received = {}
        
        # Subscribers list
        self.agent_subs = []
        
        # Initialize agent states
        for i in range(self.num_agents):
            self.agent_states[i] = {'position': np.zeros(2), 'target': np.zeros(2)}
            self.last_received[i] = time.time()
            
            # Create subscription
            sub = self.create_subscription(
                Float64MultiArray,
                f"/agent_{i}/state_vis",
                self.create_callback(i),
                10
            )
            self.agent_subs.append(sub)
        
        # MarkerArray publisher
        self.marker_pub = self.create_publisher(MarkerArray, "/aggregative_markers", 10)
        
        # Timer for visualization update
        self.timer = self.create_timer(0.1, self.update_markers)

    def create_callback(self, agent_id):
        def callback(msg):
            try:
                if len(msg.data) < 5:
                    return
                
                # [agent_id, z_x, z_y, target_x, target_y]
                self.agent_states[agent_id]['position'] = msg.data[1:3]
                self.agent_states[agent_id]['target'] = msg.data[3:5]
                self.last_received[agent_id] = time.time()
            except Exception as e:
                self.get_logger().error(f"Callback error: {str(e)}")
        return callback

    def update_markers(self):
        try:
            marker_array = MarkerArray()
            now = time.time()
            
            # Robot markers (blue spheres)
            for i in range(self.num_agents):
                if now - self.last_received[i] > 5.0:
                    continue
                
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"robot_{i}"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                pos = self.agent_states[i]['position']
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = 0.0
                
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)
            
            # Target markers (red cubes)
            for i in range(self.num_agents):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"target_{i}"
                marker.id = i + self.num_agents
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                target = self.agent_states[i]['target']
                marker.pose.position.x = float(target[0])
                marker.pose.position.y = float(target[1])
                marker.pose.position.z = 0.0
                
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)
            
            # Robot-Target connections (orange lines)
            for i in range(self.num_agents):
                if now - self.last_received[i] > 5.0:
                    continue
                
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"target_line_{i}"
                marker.id = i + 2 * self.num_agents
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.08
                
                start_point = Point()
                pos = self.agent_states[i]['position']
                start_point.x = float(pos[0])
                start_point.y = float(pos[1])
                start_point.z = 0.0
                
                end_point = Point()
                target = self.agent_states[i]['target']
                end_point.x = float(target[0])
                end_point.y = float(target[1])
                end_point.z = 0.0
                
                marker.points.append(start_point)
                marker.points.append(end_point)
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.7
                marker_array.markers.append(marker)
            
            # Barycenter marker (green sphere)
            positions = []
            for i in range(self.num_agents):
                if now - self.last_received[i] <= 5.0:
                    positions.append(self.agent_states[i]['position'])
        
            if positions:
                barycenter = np.mean(np.array(positions), axis=0)
                
                bary_marker = Marker()
                bary_marker.header.frame_id = "world"
                bary_marker.header.stamp = self.get_clock().now().to_msg()
                bary_marker.ns = "barycenter"
                bary_marker.id = 3 * self.num_agents + 2
                bary_marker.type = Marker.SPHERE
                bary_marker.action = Marker.ADD
                bary_marker.pose.position.x = float(barycenter[0])
                bary_marker.pose.position.y = float(barycenter[1])
                bary_marker.pose.position.z = 0.0
                bary_marker.scale.x = 0.4
                bary_marker.scale.y = 0.4
                bary_marker.scale.z = 0.4
                bary_marker.color.r = 0.0
                bary_marker.color.g = 1.0
                bary_marker.color.b = 0.0
                bary_marker.color.a = 1.0
                marker_array.markers.append(bary_marker)
            
            # Publish markers
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error in update_markers: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    visualizer = AggregativeVisualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
