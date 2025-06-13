#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Transform, Vector3, Quaternion
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
import struct
import copy
from scipy.spatial.transform import Rotation

class AlignmentAndPoseRegistrationNode(Node):
    def __init__(self):
        super().__init__('azure_kinect_register_pose')

        # Subscribers
        self.create_subscription(PointCloud2, '/source_scanned_point_cloud', self.scan_callback, 10)
        self.create_subscription(PointCloud2, '/canonical_model_point_cloud', self.model_callback, 10)

        # Publishers
        self.aligned_pub = self.create_publisher(PointCloud2, '/aligned_canonical_point_cloud', 10)
        self.registered_pub = self.create_publisher(PointCloud2, '/pose_registered_canonical_point_cloud', 10)

        # Storage
        self.scan_pcd = None
        self.model_pcd = None

        # Registration parameters
        self.voxel_size = 0.005  # Adjust based on your point cloud scale
        self.registration_threshold = 0.2  # Minimum fitness for successful registration
        self.use_pose_registration = True  # Flag to enable/disable pose registration

        self.get_logger().info("Alignment and Pose Registration node ready.")

    def scan_callback(self, msg):
        self.scan_pcd = self.ros2_to_o3d(msg)
        self.try_align_and_register()

    def model_callback(self, msg):
        self.model_pcd = self.ros2_to_o3d(msg)
        self.try_align_and_register()

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

    def preprocess_for_registration(self, pcd):
        """Preprocess point cloud for registration"""
        if len(pcd.points) == 0:
            return None, None

        # Remove outliers
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd_clean.points) == 0:
            return None, None

        # Downsample
        pcd_down = pcd_clean.voxel_down_sample(self.voxel_size)
        if len(pcd_down.points) == 0:
            return None, None

        # Estimate normals
        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        # Compute FPFH features
        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )

        return pcd_down, pcd_fpfh

    def perform_global_registration(self, source_down, source_fpfh, target_down, target_fpfh):
        """Perform global registration using RANSAC"""
        distance_threshold = self.voxel_size * 1.5
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,  # ransac_n
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return result

    def refine_registration(self, source, target, ransac_result):
        """Refine registration using ICP"""
        distance_threshold = self.voxel_size * 0.4
        
        # Ensure both point clouds have normals for Point-to-Plane ICP
        if not source.has_normals():
            radius_normal = self.voxel_size * 2
            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )
        
        if not target.has_normals():
            radius_normal = self.voxel_size * 2
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )
        
        # Try Point-to-Plane ICP first
        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                max_correspondence_distance=distance_threshold,
                init=ransac_result.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        except Exception as e:
            self.get_logger().warn(f"Point-to-Plane ICP failed: {str(e)}, falling back to Point-to-Point ICP")
            # Fallback to Point-to-Point ICP
            result = o3d.pipelines.registration.registration_icp(
                source, target,
                max_correspondence_distance=distance_threshold,
                init=ransac_result.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        
        return result

    def perform_pose_registration(self, model_pcd, scan_pcd):
        """Perform pose registration between model and scan"""
        try:
            # Preprocess both point clouds
            model_down, model_fpfh = self.preprocess_for_registration(model_pcd)
            scan_down, scan_fpfh = self.preprocess_for_registration(scan_pcd)

            if model_down is None or scan_down is None or model_fpfh is None or scan_fpfh is None:
                self.get_logger().warn("Failed to preprocess point clouds for registration")
                return None, False

            self.get_logger().info(f"Registration preprocessing complete: model={len(model_down.points)}, scan={len(scan_down.points)} points")

            # Global registration (RANSAC)
            result_ransac = self.perform_global_registration(
                model_down, model_fpfh, scan_down, scan_fpfh
            )
            
            self.get_logger().info(f"Global registration fitness: {result_ransac.fitness:.4f}")

            if result_ransac.fitness < 0.1:  # Very low fitness, likely failed
                self.get_logger().warn("Global registration failed - low fitness")
                return None, False

            # Refine with ICP using original (non-downsampled) point clouds for better precision
            # But first ensure they have normals
            model_copy = copy.deepcopy(model_pcd)
            scan_copy = copy.deepcopy(scan_pcd)
            
            # Estimate normals if not present
            if not model_copy.has_normals():
                radius_normal = self.voxel_size * 2
                model_copy.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
                )
            
            if not scan_copy.has_normals():
                radius_normal = self.voxel_size * 2
                scan_copy.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
                )
            
            result_icp = self.refine_registration(model_copy, scan_copy, result_ransac)
            self.get_logger().info(f"ICP registration fitness: {result_icp.fitness:.4f}")

            # Check if registration was successful
            success = result_icp.fitness > self.registration_threshold
            
            if success:
                self.get_logger().info("Pose registration successful!")
                return result_icp.transformation, True
            else:
                self.get_logger().warn(f"Pose registration failed - fitness {result_icp.fitness:.4f} below threshold {self.registration_threshold}")
                return None, False

        except Exception as e:
            self.get_logger().error(f"Pose registration error: {str(e)}")
            return None, False

    def transformation_to_pose_msg(self, transformation):
        """Convert 4x4 transformation matrix to ROS Transform message"""
        # Extract translation
        translation = Vector3()
        translation.x = float(transformation[0, 3])
        translation.y = float(transformation[1, 3])
        translation.z = float(transformation[2, 3])

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = transformation[:3, :3]
        r = Rotation.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns [x, y, z, w]

        quaternion = Quaternion()
        quaternion.x = float(quat[0])
        quaternion.y = float(quat[1])
        quaternion.z = float(quat[2])
        quaternion.w = float(quat[3])

        transform = Transform()
        transform.translation = translation
        transform.rotation = quaternion

        return transform

    def try_align_and_register(self):
        if self.scan_pcd is None or self.model_pcd is None:
            return

        self.get_logger().info("Both point clouds received. Processing alignment, scaling, and pose registration.")

        # Make copies to preserve original data
        working_model_pcd = copy.deepcopy(self.model_pcd)
        working_scan_pcd = copy.deepcopy(self.scan_pcd)

        # Step 1: Basic Alignment and Scaling (from original code)
        # Compute centroids
        scan_centroid = np.mean(np.asarray(working_scan_pcd.points), axis=0)
        model_centroid = np.mean(np.asarray(working_model_pcd.points), axis=0)

        # Find characteristic feature points
        scan_points = np.asarray(working_scan_pcd.points)
        model_points = np.asarray(working_model_pcd.points)

        scan_feature = scan_points[np.argmax(scan_points[:, 1])]  # Y-axis for scan
        model_feature = model_points[np.argmax(model_points[:, 2])]  # Z-axis for model

        # Orientation vectors
        scan_vector = scan_feature - scan_centroid
        model_vector = model_feature - model_centroid

        # Align orientation vectors
        R = self.compute_rotation_matrix(model_vector, scan_vector)
        working_model_pcd.rotate(R, center=model_centroid)

        # Scale canonical model to match scan bounding box
        scan_bbox = working_scan_pcd.get_axis_aligned_bounding_box()
        model_bbox = working_model_pcd.get_axis_aligned_bounding_box()

        scale_factors = scan_bbox.get_extent() / model_bbox.get_extent()
        uniform_scale = np.min(scale_factors)

        working_model_pcd.scale(uniform_scale, center=model_centroid)

        # Move model centroid to scan centroid
        aligned_model_centroid = np.mean(np.asarray(working_model_pcd.points), axis=0)
        translation = scan_centroid - aligned_model_centroid
        working_model_pcd.translate(translation)

        # Publish aligned canonical model
        aligned_msg = self.o3d_to_ros2(working_model_pcd, frame_id="map")
        self.aligned_pub.publish(aligned_msg)
        self.get_logger().info("Published aligned model.")

        # Step 2: Pose Registration (if enabled)
        if self.use_pose_registration and len(working_model_pcd.points) > 100 and len(working_scan_pcd.points) > 100:
            transformation, success = self.perform_pose_registration(working_model_pcd, working_scan_pcd)
            
            if success and transformation is not None:
                # Apply pose registration transformation
                registered_model_pcd = copy.deepcopy(working_model_pcd)
                registered_model_pcd.transform(transformation)

                # Publish pose-registered canonical model
                registered_msg = self.o3d_to_ros2(registered_model_pcd, frame_id="map")
                self.registered_pub.publish(registered_msg)
                self.get_logger().info("Published pose-registered canonical point cloud.")

                # Log pose information
                pose_msg = self.transformation_to_pose_msg(transformation)
                self.get_logger().info(f"Estimated pose - Translation: [{pose_msg.translation.x:.3f}, {pose_msg.translation.y:.3f}, {pose_msg.translation.z:.3f}]")
                self.get_logger().info(f"Estimated pose - Rotation: [{pose_msg.rotation.x:.3f}, {pose_msg.rotation.y:.3f}, {pose_msg.rotation.z:.3f}, {pose_msg.rotation.w:.3f}]")
            else:
                self.get_logger().warn("Pose registration failed, fallback to aligned.")
                # Still publish the aligned model as fallback
                self.registered_pub.publish(aligned_msg)
        else:
            if not self.use_pose_registration:
                self.get_logger().info("Pose registration disabled - publishing aligned model only")
            else:
                self.get_logger().warn(f"Insufficient points for pose registration: model={len(working_model_pcd.points)}, scan={len(working_scan_pcd.points)}")
            
            # Publish aligned model as registered output when pose registration is not performed
            self.registered_pub.publish(aligned_msg)

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
    node = AlignmentAndPoseRegistrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()