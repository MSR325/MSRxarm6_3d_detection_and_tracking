import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray
import numpy as np
from sensor_msgs_py import point_cloud2

class PointCloudScaler(Node):
    def __init__(self):
        super().__init__('point_cloud_scaler')
        
        # Parameters
        self.declare_parameter('input_topic', '/canonical_model')
        self.declare_parameter('output_topic', '/canonical_model_scaled')
        
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        
        # Current scaling factors
        self.current_scale = [0.01, 0.01, 0.01]
        self.latest_pointcloud = None
        
        # Subscriber for original point cloud
        self.pc_subscriber = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10
        )
        
        # Subscriber for transform parameters
        self.transform_sub = self.create_subscription(
            Float64MultiArray,
            '/transform_params',
            self.transform_params_callback,
            10
        )
        
        # Publisher for scaled point cloud
        self.pc_publisher = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )
        
        self.get_logger().info(f"Point cloud scaler started")
        self.get_logger().info(f"Input: {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
    
    def pointcloud_callback(self, msg):
        """Store the latest point cloud"""
        self.latest_pointcloud = msg
        self.publish_scaled_pointcloud()
    
    def transform_params_callback(self, msg):
        """Update scaling parameters"""
        if len(msg.data) >= 9:
            new_scale = [msg.data[6], msg.data[7], msg.data[8]]
            if new_scale != self.current_scale:
                self.current_scale = new_scale
                self.get_logger().info(
                    f"Updated scale: {self.current_scale}",
                    throttle_duration_sec=1.0
                )
                # Re-publish with new scale
                if self.latest_pointcloud is not None:
                    self.publish_scaled_pointcloud()
    
    def publish_scaled_pointcloud(self):
        """Apply scaling and publish the point cloud"""
        if self.latest_pointcloud is None:
            return
        
        try:
            # Extract points from point cloud message
            points = list(point_cloud2.read_points(
                self.latest_pointcloud,
                field_names=("x", "y", "z"),
                skip_nans=True
            ))
            
            if not points:
                return
            
            # Convert to numpy array and apply scaling
            points_array = np.array(points)
            scaled_points = points_array * np.array(self.current_scale)
            
            # Create new point cloud message
            scaled_msg = point_cloud2.create_cloud_xyz32(
                header=self.latest_pointcloud.header,
                points=scaled_points
            )
            
            # Update timestamp
            scaled_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Publish scaled point cloud
            self.pc_publisher.publish(scaled_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error scaling point cloud: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PointCloudScaler()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()