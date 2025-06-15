import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
import struct

class CanonicalModelPublisher(Node):
    def __init__(self, model_path):
        super().__init__('inteld435i_source_canonical_point_cloud_node')
        
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
        self.get_logger().info("Canonical model publisher started")
        
    def load_point_cloud(self):
        try:
            pcd = o3d.io.read_point_cloud(self.model_path)
            if len(pcd.points) == 0:
                self.get_logger().error("Loaded empty point cloud!")
                return None
            
            # Downsample for better performance
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            return pcd
            
        except Exception as e:
            self.get_logger().error(f"Error loading point cloud: {str(e)}")
            return None
    
    def publish_point_cloud(self):
        if self.point_cloud is None:
            return
        
        points = np.asarray(self.point_cloud.points)
        colors = np.asarray(self.point_cloud.colors) if self.point_cloud.has_colors() else None
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_link"
        
        # Create fields array
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Prepare point data
        if colors is not None:
            # Convert colors to uint8 and pack into single float32
            colors_uint8 = (colors * 255).astype(np.uint8)
            rgb_packed = np.array([
                struct.unpack('f', struct.pack('BBBB', b, g, r, 0))[0]
                for r, g, b in colors_uint8
            ], dtype=np.float32)
            
            fields.append(
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
            )
            point_data = np.array([(p[0], p[1], p[2], c) for p, c in zip(points, rgb_packed)],
                                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        else:
            point_data = np.array([(p[0], p[1], p[2]) for p in points],
                                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        
        # Create PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, point_data)
        cloud_msg.is_dense = True
        
        self.publisher.publish(cloud_msg)
        self.get_logger().info(f"Published {len(points)} points", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    
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