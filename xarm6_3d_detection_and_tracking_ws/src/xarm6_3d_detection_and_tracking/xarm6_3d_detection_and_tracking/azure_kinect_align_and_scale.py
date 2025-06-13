#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
import struct

class AlignmentAndScalingNode(Node):
    def __init__(self):
        super().__init__('azure_kinect_align_and_scale')

        # Subscribers
        self.create_subscription(PointCloud2, '/source_scanned_point_cloud', self.scan_callback, 10)
        self.create_subscription(PointCloud2, '/canonical_model_point_cloud', self.model_callback, 10)

        # Publisher
        self.aligned_pub = self.create_publisher(PointCloud2, '/aligned_canonical_point_cloud', 10)

        # Storage
        self.scan_pcd = None
        self.model_pcd = None

        self.get_logger().info("Alignment and Scaling node ready.")

    def scan_callback(self, msg):
        self.scan_pcd = self.ros2_to_o3d(msg)
        self.try_align_and_publish()

    def model_callback(self, msg):
        self.model_pcd = self.ros2_to_o3d(msg)
        self.try_align_and_publish()

    def compute_rotation_matrix(self, v_from, v_to):
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        cross_prod = np.cross(v_from, v_to)
        norm_cross = np.linalg.norm(cross_prod)

        if norm_cross < 1e-6:
            return np.eye(3)

        cross_prod = cross_prod / norm_cross
        angle = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))

        K = np.array([
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def try_align_and_publish(self):
        if self.scan_pcd is None or self.model_pcd is None:
            return

        self.get_logger().info("Both point clouds received. Performing alignment and scaling.")

        # Compute centroids
        scan_centroid = np.mean(np.asarray(self.scan_pcd.points), axis=0)
        model_centroid = np.mean(np.asarray(self.model_pcd.points), axis=0)

        # Find characteristic feature points
        scan_points = np.asarray(self.scan_pcd.points)
        model_points = np.asarray(self.model_pcd.points)

        scan_feature = scan_points[np.argmax(scan_points[:, 1])]  # Y-axis for scan
        print(f"Scan feature: {scan_feature}")
        model_feature = model_points[np.argmax(model_points[:, 2])]  # Z-axis for model
        print(f"Model feature: {model_feature}")

        # Orientation vectors
        scan_vector = scan_feature - scan_centroid
        model_vector = model_feature - model_centroid

        # Align orientation vectors
        R = self.compute_rotation_matrix(model_vector, scan_vector)
        self.model_pcd.rotate(R, center=model_centroid)

        # Scale canonical model to match scan bounding box
        scan_bbox = self.scan_pcd.get_axis_aligned_bounding_box()
        model_bbox = self.model_pcd.get_axis_aligned_bounding_box()

        scale_factors = scan_bbox.get_extent() / model_bbox.get_extent()
        uniform_scale = np.min(scale_factors)

        self.model_pcd.scale(uniform_scale, center=model_centroid)

        # Move model centroid to scan centroid
        aligned_model_centroid = np.mean(np.asarray(self.model_pcd.points), axis=0)
        translation = scan_centroid - aligned_model_centroid
        self.model_pcd.translate(translation)

        # Publish aligned canonical model
        aligned_msg = self.o3d_to_ros2(self.model_pcd, frame_id="map")
        self.aligned_pub.publish(aligned_msg)

        self.get_logger().info("Published aligned canonical point cloud.")

        # Clear to avoid redundant alignment
        self.scan_pcd = None
        self.model_pcd = None

    def ros2_to_o3d(self, cloud_msg):
        """Convert ROS2 PointCloud2 to Open3D point cloud"""
        try:
            # Get available fields
            field_names = [f.name for f in cloud_msg.fields]
            
            # Read points
            points_gen = point_cloud2.read_points(cloud_msg, skip_nans=True)
            
            points = []
            colors = []
            
            for p in points_gen:
                points.append([p[field_names.index('x')],
                            p[field_names.index('y')],
                            p[field_names.index('z')]])
                
                if 'rgb' in field_names:
                    packed = p[field_names.index('rgb')]
                    # Unpack RGB from float32
                    rgb_bytes = struct.pack('f', packed)
                    b, g, r = rgb_bytes[0], rgb_bytes[1], rgb_bytes[2]
                    colors.append([r/255.0, g/255.0, b/255.0])
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            
            if colors:
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            return pcd
            
        except Exception as e:
            self.get_logger().error(f"Error converting PointCloud2: {str(e)}")
            return None

    def unpack_rgb(self, packed_rgb):
        """Convert packed float32 RGB to 0-1 range"""
        b = packed_rgb & 0xFF
        g = (packed_rgb >> 8) & 0xFF
        r = (packed_rgb >> 16) & 0xFF
        return np.array([r, g, b]) / 255.0

    def o3d_to_ros2(self, o3d_cloud, frame_id="map"):
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
            data.extend(struct.pack('fff', points[i][0], points[i][1], points[i][2]))
            if colors.shape[0] > 0:
                r, g, b = (colors[i] * 255).astype(np.uint8)
            else:
                r, g, b = 255, 255, 255  # default white if no color info
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
 
def main(args=None):
    rclpy.init(args=args)
    node = AlignmentAndScalingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()