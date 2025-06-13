import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Transform, Vector3, Quaternion
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

class CanonicalModelPublisher(Node):
    def __init__(self, model_path):
        super().__init__('azure_kinect_target_canonical_point_cloud')
        
        # Load the point cloud
        self.model_path = model_path
        self.point_cloud = self.load_point_cloud()
        
        if self.point_cloud is None:
            self.get_logger().error("Failed to load point cloud!")
            return
        
        # Create publisher
        self.publisher = self.create_publisher(
            PointCloud2,
            '/canonical_model_point_cloud',  # Topic name
            10  # Queue size
        )
        
        # Publish at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_point_cloud)
        self.get_logger().info(f"Canonical model publisher started. Publishing to /canonical_model_point_cloud")
    
    def load_point_cloud(self):
        """Load and preprocess the point cloud"""
        try:
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(self.model_path)
            if len(pcd.points) == 0:
                self.get_logger().error("Loaded empty point cloud!")
                return None
            
            # Basic preprocessing
            pcd = pcd.voxel_down_sample(voxel_size=0.005)  # Downsample
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # Remove outliers
            
            self.get_logger().info(f"Loaded point cloud with {len(pcd.points)} points")
            return pcd
            
        except Exception as e:
            self.get_logger().error(f"Error loading point cloud: {str(e)}")
            return None
    
    def publish_point_cloud(self):
        """Convert Open3D point cloud to ROS2 message and publish"""
        if self.point_cloud is None:
            return
        
        # Convert to numpy arrays
        points = np.asarray(self.point_cloud.points)
        colors = np.asarray(self.point_cloud.colors) if self.point_cloud.has_colors() else None
        
        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"  # Change this to your desired frame
        
        # Create fields
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1)
        ]
        
        # Add color fields if available
        if colors is not None:
            fields.append(point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1))
        
        # Convert colors to packed RGB format if they exist
        if colors is not None:
            # Scale colors to 0-255 and pack into single float32
            colors_uint8 = (colors * 255).astype(np.uint8)
            packed_colors = (colors_uint8[:, 0] << 16) | (colors_uint8[:, 1] << 8) | colors_uint8[:, 2]
            points_with_colors = np.hstack((points, packed_colors.reshape(-1, 1)))
        else:
            points_with_colors = points
        
        # Create PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, points_with_colors)
        
        # Publish
        self.publisher.publish(cloud_msg)
        self.get_logger().info(f"Published point cloud with {len(points)} points", throttle_duration_sec=5.0)

def main(args=None):
    rclpy.init(args=args)
    
    # Path to your canonical model - modify this path
    canonical_model_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"  # Can be .ply, .pcd, etc.
    
    # You can also pass this as a command line argument
    import sys
    if len(sys.argv) > 1:
        canonical_model_path = sys.argv[1]
    
    try:
        node = CanonicalModelPublisher(canonical_model_path)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to start node: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()