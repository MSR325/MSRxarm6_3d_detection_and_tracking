#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
import struct
import copy

class OilPanPoseRegistrationNode(Node):
    def __init__(self):
        super().__init__('intel453i_register_pose_point_clouds')

        # Subscribers
        self.create_subscription(PointCloud2, '/scanned_model_point_cloud', self.scan_callback, 10)
        self.create_subscription(PointCloud2, '/aligned_canonical_model_point_cloud', self.aligned_model_callback, 10)

        # Publishers
        self.ransac_pub = self.create_publisher(PointCloud2, '/ransac_registered_point_cloud', 10)
        self.icp_pub = self.create_publisher(PointCloud2, '/icp_registered_point_cloud', 10)

        # Storage
        self.scan_pcd = None
        self.aligned_model_pcd = None
        
        # Registration parameters
        self.voxel_size = 0.003  # 5mm voxel size - adjust based on your oil pan size
        
        self.get_logger().info("Oil Pan Pose Registration node ready.")
        self.get_logger().info(f"Using voxel size: {self.voxel_size}")

    def scan_callback(self, msg):
        self.scan_pcd = self.ros2_to_o3d(msg)
        if self.scan_pcd is not None:
            self.get_logger().info(f"Received scan point cloud with {len(self.scan_pcd.points)} points")
        self.try_register_and_publish()

    def aligned_model_callback(self, msg):
        self.aligned_model_pcd = self.ros2_to_o3d(msg)
        if self.aligned_model_pcd is not None:
            self.get_logger().info(f"Received aligned model point cloud with {len(self.aligned_model_pcd.points)} points")
        self.try_register_and_publish()

    def preprocess_point_cloud(self, pcd, voxel_size):
        """Preprocess point cloud: downsample, estimate normals, compute FPFH features"""
        self.get_logger().info(f":: Downsample with a voxel size of {voxel_size:.3f}")
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        if len(pcd_down.points) == 0:
            self.get_logger().error("Downsampled point cloud is empty!")
            return None, None
        
        self.get_logger().info(f":: Downsampled to {len(pcd_down.points)} points")
        
        # Estimate normals
        radius_normal = voxel_size * 2
        self.get_logger().info(f":: Estimate normals with search radius {radius_normal:.3f}")
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        
        # Compute FPFH features
        radius_feature = voxel_size * 5
        self.get_logger().info(f":: Compute FPFH feature with search radius {radius_feature:.3f}")
        try:
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )
            self.get_logger().info(f":: FPFH features computed: {len(pcd_fpfh.data[0])} features")
        except Exception as e:
            self.get_logger().error(f"Error computing FPFH features: {str(e)}")
            return pcd_down, None
        
        return pcd_down, pcd_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        """Execute RANSAC-based global registration"""
        if source_fpfh is None or target_fpfh is None:
            self.get_logger().error("FPFH features are None, cannot perform RANSAC registration")
            return None
            
        distance_threshold = voxel_size * 1.5
        self.get_logger().info(":: RANSAC registration on downsampled point clouds")
        self.get_logger().info(f"   Using distance threshold {distance_threshold:.3f}")
        
        try:
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, 
                True,  # mutual_filter
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,  # ransac_n
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            
            self.get_logger().info(f":: RANSAC registration result:")
            self.get_logger().info(f"   Fitness: {result.fitness:.6f}")
            self.get_logger().info(f"   RMSE: {result.inlier_rmse:.6f}")
            self.get_logger().info(f"   Correspondences: {len(result.correspondence_set)}")
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"Error during RANSAC registration: {str(e)}")
            return None

    def refine_registration(self, source, target, initial_transformation, voxel_size):
        """Refine registration using Point-to-Plane ICP"""
        distance_threshold = voxel_size * 0.4
        self.get_logger().info(":: Point-to-plane ICP registration")
        self.get_logger().info(f"   Using strict distance threshold {distance_threshold:.3f}")
        
        # Ensure normals are computed for both point clouds
        self.compute_normals(source, voxel_size * 2)
        self.compute_normals(target, voxel_size * 2)
        
        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target, 
                distance_threshold, 
                initial_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )
            
            self.get_logger().info(f":: ICP registration result:")
            self.get_logger().info(f"   Fitness: {result.fitness:.6f}")
            self.get_logger().info(f"   RMSE: {result.inlier_rmse:.6f}")
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"Error during ICP registration: {str(e)}")
            return None

    def compute_normals(self, pcd, radius):
        """Compute normals for point cloud"""
        try:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
        except Exception as e:
            self.get_logger().error(f"Error computing normals: {str(e)}")

    def try_register_and_publish(self):
        if self.scan_pcd is None or self.aligned_model_pcd is None:
            return

        self.get_logger().info("Both point clouds received. Starting pose registration pipeline.")

        try:
            # Make copies to avoid modifying originals
            source = copy.deepcopy(self.aligned_model_pcd)  # This is our aligned canonical model
            target = copy.deepcopy(self.scan_pcd)           # This is the scanned oil pan

            # Preprocess both point clouds
            self.get_logger().info("Preprocessing source (aligned model) point cloud...")
            source_down, source_fpfh = self.preprocess_point_cloud(source, self.voxel_size)
            
            self.get_logger().info("Preprocessing target (scan) point cloud...")
            target_down, target_fpfh = self.preprocess_point_cloud(target, self.voxel_size)

            if source_down is None or target_down is None:
                self.get_logger().error("Preprocessing failed, cannot continue registration")
                return

            # Execute RANSAC global registration
            self.get_logger().info("Executing RANSAC global registration...")
            result_ransac = self.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, self.voxel_size
            )

            if result_ransac is None:
                self.get_logger().error("RANSAC registration failed")
                return

            # Apply RANSAC transformation and publish
            source_ransac = copy.deepcopy(source)
            source_ransac.transform(result_ransac.transformation)
            
            # Color the RANSAC result (orange)
            source_ransac.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
            
            ransac_msg = self.o3d_to_ros2(source_ransac, frame_id="camera_link")
            self.ransac_pub.publish(ransac_msg)
            self.get_logger().info("Published RANSAC registration result")

            # Refine with ICP
            self.get_logger().info("Refining registration with ICP...")
            result_icp = self.refine_registration(
                source, target, result_ransac.transformation, self.voxel_size
            )

            if result_icp is None:
                self.get_logger().error("ICP registration failed")
                return

            # Apply ICP transformation and publish
            source_icp = copy.deepcopy(source)
            source_icp.transform(result_icp.transformation)
            
            # Color the ICP result (cyan)
            source_icp.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
            
            icp_msg = self.o3d_to_ros2(source_icp, frame_id="camera_link")
            self.icp_pub.publish(icp_msg)
            self.get_logger().info("Published ICP registration result")

            # Log final transformation matrix
            self.get_logger().info("Final ICP transformation matrix:")
            for i, row in enumerate(result_icp.transformation):
                self.get_logger().info(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        except Exception as e:
            self.get_logger().error(f"Error during registration pipeline: {str(e)}")

        # Clear to avoid redundant processing
        self.scan_pcd = None
        self.aligned_model_pcd = None

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

    def o3d_to_ros2(self, o3d_cloud, frame_id="map"):
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
                    r, g, b = 255, 255, 255  # Default white
                
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
    node = OilPanPoseRegistrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()