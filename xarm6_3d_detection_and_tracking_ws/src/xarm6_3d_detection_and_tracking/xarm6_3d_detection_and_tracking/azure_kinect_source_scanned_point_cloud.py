import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import open3d as o3d
import numpy as np
import struct

class CanonicalModelPublisher(Node):
    def __init__(self, model_path):
        super().__init__('canonical_model_publisher')
        
        self.model_path = model_path
        self.point_cloud = self.load_point_cloud()
        
        if self.point_cloud is None:
            self.get_logger().error("Failed to load point cloud!")
            return
        
        self.publisher = self.create_publisher(
            PointCloud2,
            '/canonical_model_point_cloud',
            10
        )
        
        self.timer = self.create_timer(1.0, self.publish_point_cloud)
        self.get_logger().info("Canonical model publisher ready")

    def load_point_cloud(self):
        try:
            pcd = o3d.io.read_point_cloud(self.model_path)
            if len(pcd.points) == 0:
                self.get_logger().error("Empty point cloud loaded!")
                return None
            
            # Basic preprocessing
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Ensure colors exist (set to white if not present)
            if not pcd.has_colors():
                colors = np.ones((len(pcd.points), 3)) * 0.8  # Light gray
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            self.get_logger().info(f"Loaded model with {len(pcd.points)} points")
            return pcd
            
        except Exception as e:
            self.get_logger().error(f"Error loading point cloud: {str(e)}")
            return None

    def convert_o3d_to_ros2_pointcloud2(self, o3d_cloud, frame_id="map"):
        """Your proven working conversion function"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        points = np.asarray(o3d_cloud.points)
        colors = np.asarray(o3d_cloud.colors)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        point_step = 16
        data = bytearray()

        for i in range(len(points)):
            # Pack position
            data.extend(struct.pack('fff', points[i][0], points[i][1], points[i][2]))
            
            # Pack color (RGB32 format)
            r, g, b = (colors[i] * 255).astype(np.uint8)
            rgb = (r << 16) | (g << 8) | b
            data.extend(struct.pack('I', rgb))

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = point_step
        cloud_msg.row_step = point_step * len(points)
        cloud_msg.is_dense = True
        cloud_msg.data = data

        return cloud_msg

    def publish_point_cloud(self):
        if self.point_cloud is None:
            return
            
        cloud_msg = self.convert_o3d_to_ros2_pointcloud2(self.point_cloud)
        self.publisher.publish(cloud_msg)
        self.get_logger().info("Published point cloud", throttle_duration_sec=5.0)

def main(args=None):
    rclpy.init(args=args)
    
    # Update this path to your canonical model
    model_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"
    
    node = CanonicalModelPublisher(model_path)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()