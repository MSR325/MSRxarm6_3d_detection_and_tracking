#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
import struct

class OilPanAlignmentAndScalingNode(Node):
    def __init__(self):
        super().__init__('inteld435i_align_and_scale_point_clouds')

        # Subscribers
        self.create_subscription(PointCloud2, '/scanned_model_point_cloud', self.scan_callback, 10)
        self.create_subscription(PointCloud2, '/canonical_model_point_cloud', self.model_callback, 10)

        # Publisher
        self.aligned_pub = self.create_publisher(PointCloud2, '/aligned_canonical_model_point_cloud', 10)

        # Storage
        self.scan_pcd = None
        self.model_pcd = None

        self.get_logger().info("Oil Pan Alignment and Scaling node ready.")

    def scan_callback(self, msg):
        self.scan_pcd = self.ros2_to_o3d(msg)
        if self.scan_pcd is not None:
            self.get_logger().info(f"Received scan point cloud with {len(self.scan_pcd.points)} points")
        self.try_align_and_publish()

    def model_callback(self, msg):
        self.model_pcd = self.ros2_to_o3d(msg)
        if self.model_pcd is not None:
            self.get_logger().info(f"Received model point cloud with {len(self.model_pcd.points)} points")
        self.try_align_and_publish()

    def identify_oil_pan_drain_plug(self, pcd):
        """Identify the drain plug (lowest point) of the oil pan"""
        points = np.asarray(pcd.points)
        z_coordinates = points[:, 2]
        # Find the lowest point (drain plug area)
        min_z_idx = np.argmin(z_coordinates)
        return points[min_z_idx]

    def identify_oil_pan_rim_center(self, pcd):
        """Identify the center of the oil pan rim (highest points)"""
        points = np.asarray(pcd.points)
        z_coordinates = points[:, 2]
        # Get top 5% of points to find rim
        threshold = np.percentile(z_coordinates, 95)
        rim_points = points[z_coordinates > threshold]
        # Return the centroid of rim points
        return np.mean(rim_points, axis=0)

    def identify_oil_pan_drain_plug_scan(self, pcd):
        """Identify the drain plug for scanned oil pan (might be oriented differently)"""
        points = np.asarray(pcd.points)
        
        # For scanned data, the drain plug might be the furthest point in any direction
        # Try finding extreme points in all axes
        centroids = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroids, axis=1)
        extreme_idx = np.argmax(distances)
        
        return points[extreme_idx]

    def identify_oil_pan_rim_center_scan(self, pcd):
        """Identify the rim center for scanned oil pan"""
        points = np.asarray(pcd.points)
        
        # For oil pan, rim is typically the collection of points furthest from the centroid
        # in the horizontal plane
        centroid = np.mean(points, axis=0)
        
        # Calculate distances in XY plane only (assuming Z is up)
        xy_distances = np.sqrt((points[:, 0] - centroid[0])**2 + (points[:, 1] - centroid[1])**2)
        threshold = np.percentile(xy_distances, 90)
        rim_points = points[xy_distances > threshold]
        
        return np.mean(rim_points, axis=0)

    def compute_oil_pan_orientation_vector(self, pcd, is_scan=False):
        """Compute orientation vector from rim center to drain plug"""
        if is_scan:
            rim_center = self.identify_oil_pan_rim_center_scan(pcd)
            drain_plug = self.identify_oil_pan_drain_plug_scan(pcd)
        else:
            rim_center = self.identify_oil_pan_rim_center(pcd)
            drain_plug = self.identify_oil_pan_drain_plug(pcd)
        
        # Vector from rim center to drain plug
        direction_vector = drain_plug - rim_center
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)
        
        return normalized_vector, rim_center, drain_plug

    def compute_rotation_matrix(self, v_from, v_to):
        """Compute rotation matrix to align v_from to v_to"""
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        
        # Check if vectors are already aligned
        dot_product = np.dot(v_from, v_to)
        if dot_product > 0.9999:  # Already aligned
            return np.eye(3)
        elif dot_product < -0.9999:  # Opposite direction
            # Find a perpendicular vector
            if abs(v_from[0]) < 0.9:
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            # Create rotation by 180 degrees around perpendicular axis
            perp = perp - np.dot(perp, v_from) * v_from
            perp = perp / np.linalg.norm(perp)
            return 2 * np.outer(perp, perp) - np.eye(3)
        
        # Standard case - use Rodrigues' formula
        cross_prod = np.cross(v_from, v_to)
        cross_prod_norm = np.linalg.norm(cross_prod)
        
        if cross_prod_norm < 1e-6:
            return np.eye(3)
        
        # Skew-symmetric matrix
        K = np.array([
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0]
        ])
        
        # Rodrigues' formula
        angle = np.arcsin(cross_prod_norm)
        R = np.eye(3) + K + (K @ K) * ((1 - dot_product) / (cross_prod_norm**2))
        
        return R

    def try_align_and_publish(self):
        if self.scan_pcd is None or self.model_pcd is None:
            return

        self.get_logger().info("Both point clouds received. Performing oil pan alignment and scaling.")

        try:
            # Make copies to avoid modifying originals
            model_copy = o3d.geometry.PointCloud(self.model_pcd)
            scan_copy = o3d.geometry.PointCloud(self.scan_pcd)

            # Compute centroids
            scan_centroid = np.mean(np.asarray(scan_copy.points), axis=0)
            model_centroid = np.mean(np.asarray(model_copy.points), axis=0)

            self.get_logger().info(f"Scan centroid: {scan_centroid}")
            self.get_logger().info(f"Model centroid: {model_centroid}")

            # Move both point clouds to origin
            model_copy.translate(-model_centroid)
            scan_copy.translate(-scan_centroid)

            # Compute orientation vectors for oil pan alignment
            model_vector, model_rim, model_drain = self.compute_oil_pan_orientation_vector(model_copy, is_scan=False)
            scan_vector, scan_rim, scan_drain = self.compute_oil_pan_orientation_vector(scan_copy, is_scan=True)

            self.get_logger().info(f"Model orientation vector: {model_vector}")
            self.get_logger().info(f"Scan orientation vector: {scan_vector}")

            # Compute rotation to align model to scan orientation
            R = self.compute_rotation_matrix(model_vector, scan_vector)
            model_copy.rotate(R, center=[0, 0, 0])

            # Scale model to match scan dimensions
            scan_bbox = scan_copy.get_axis_aligned_bounding_box()
            model_bbox = model_copy.get_axis_aligned_bounding_box()

            scan_extent = scan_bbox.get_extent()
            model_extent = model_bbox.get_extent()

            self.get_logger().info(f"Scan dimensions: {scan_extent}")
            self.get_logger().info(f"Model dimensions: {model_extent}")

            # Use uniform scaling based on the largest dimension to preserve shape
            scale_factors = scan_extent / model_extent
            # Remove any zero or infinite scale factors
            valid_scales = scale_factors[np.isfinite(scale_factors) & (scale_factors > 0)]
            if len(valid_scales) > 0:
                uniform_scale = np.median(valid_scales)  # Use median for robustness
            else:
                uniform_scale = 1.0
                self.get_logger().warning("Could not compute valid scale factor, using 1.0")

            self.get_logger().info(f"Applying uniform scale: {uniform_scale}")
            model_copy.scale(uniform_scale, center=[0, 0, 0])

            # Move model to scan centroid
            model_copy.translate(scan_centroid)

            # Publish aligned model
            aligned_msg = self.o3d_to_ros2(model_copy, frame_id="camera_link")
            self.aligned_pub.publish(aligned_msg)

            self.get_logger().info("Published aligned and scaled oil pan canonical model.")

        except Exception as e:
            self.get_logger().error(f"Error during alignment: {str(e)}")

        # Clear to avoid redundant processing
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
                    try:
                        packed = p[field_names.index('rgb')]
                        if isinstance(packed, float):
                            # Convert float to bytes
                            rgb_bytes = struct.pack('f', packed)
                            b, g, r = rgb_bytes[0], rgb_bytes[1], rgb_bytes[2]
                        else:
                            # Assume it's already an integer
                            r = (packed >> 16) & 0xFF
                            g = (packed >> 8) & 0xFF
                            b = packed & 0xFF
                        colors.append([r/255.0, g/255.0, b/255.0])
                    except:
                        colors.append([1.0, 1.0, 1.0])  # Default white
            
            if not points:
                self.get_logger().warning("No valid points found in point cloud")
                return None

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            
            if colors and len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            return pcd
            
        except Exception as e:
            self.get_logger().error(f"Error converting PointCloud2: {str(e)}")
            return None

    def o3d_to_ros2(self, o3d_cloud, frame_id="camera_link"):
        """Convert Open3D point cloud to ROS2 PointCloud2"""
        try:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = frame_id

            points = np.asarray(o3d_cloud.points)
            
            # Check if colors exist
            has_colors = hasattr(o3d_cloud, 'colors') and len(o3d_cloud.colors) > 0
            colors = np.asarray(o3d_cloud.colors) if has_colors else None

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]

            point_step = 16
            data = bytearray()

            for i in range(len(points)):
                # Add XYZ coordinates
                data.extend(struct.pack('fff', points[i][0], points[i][1], points[i][2]))
                
                # Add RGB color
                if has_colors and i < len(colors):
                    r, g, b = (colors[i] * 255).astype(np.uint8)
                else:
                    r, g, b = 0, 255, 0  # Default green for aligned model
                
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
            
        except Exception as e:
            self.get_logger().error(f"Error creating ROS2 PointCloud2: {str(e)}")
            return None
 
def main(args=None):
    rclpy.init(args=args)
    node = OilPanAlignmentAndScalingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()