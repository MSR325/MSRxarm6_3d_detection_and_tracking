import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Transform, Vector3, Quaternion
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import struct
import copy
import time
import json
from scipy.spatial.transform import Rotation

class PoseRegistrationNode(Node):
    def __init__(self, canonical_model_path):
        super().__init__('pose_registration')
        self.bridge = CvBridge()
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/rgb/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 10)

        # Publishers
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/filtered_source_scanned_point_cloud', 10)
        self.registered_cloud_pub = self.create_publisher(
            PointCloud2, '/registered_point_cloud', 10)
        
        # Image storage
        self.color_image = None
        self.depth_image = None
        
        # Registration parameters
        self.min_component_area = 500
        self.voxel_size = 0.005  # Adjust based on your point cloud scale
        self.registration_threshold = 0.3  # Minimum fitness for successful registration
        
        # Load canonical model
        self.canonical_model = self.load_canonical_model(canonical_model_path)
        if self.canonical_model is None:
            self.get_logger().error("Failed to load canonical model!")
            return
        
        self.get_logger().info(f"Loaded canonical model with {len(self.canonical_model.points)} points")
        
        # Registration state
        self.last_transformation = np.identity(4)
        self.registration_success = False
        
    def load_canonical_model(self, model_path):
        """Load the canonical 3D model"""
        try:
            model = o3d.io.read_point_cloud(model_path)
            if len(model.points) == 0:
                self.get_logger().error(f"Empty point cloud loaded from {model_path}")
                return None
            
            # Preprocess canonical model
            model = self.preprocess_canonical_model(model)
            return model
        except Exception as e:
            self.get_logger().error(f"Error loading canonical model: {str(e)}")
            return None
    
    def preprocess_canonical_model(self, model):
        """Preprocess the canonical model for registration"""
        # Remove outliers
        model, _ = model.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample
        model = model.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        radius_normal = self.voxel_size * 2
        model.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        
        # Compute FPFH features
        radius_feature = self.voxel_size * 5
        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            model, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        # Store FPFH features as an attribute
        model.fpfh = model_fpfh
        
        self.get_logger().info(f"Preprocessed canonical model: {len(model.points)} points")
        return model
    
    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_register()
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_register()
    
    def create_color_mask(self, color_image):
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lo_black = np.array([0, 0, 0])
        up_black = np.array([180, 255, 100])
        color_mask = cv2.inRange(hsv_img, lo_black, up_black)
        return color_mask
    
    def create_depth_mask(self, depth_image, target_shape):
        lo_d = 200
        up_d = 1200
        depth_mask = cv2.inRange(depth_image, lo_d, up_d)
        return depth_mask
    
    def combine_masks(self, depth_mask, color_mask):
        return cv2.bitwise_and(depth_mask, color_mask)
    
    def clean_mask_with_connected_components(self, combined_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        clean_mask = np.zeros_like(combined_mask)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            if stats[largest_idx, cv2.CC_STAT_AREA] >= self.min_component_area:
                clean_mask[labels == largest_idx] = 255

        return clean_mask
    
    def generate_rgbd_image(self, combined_mask, depth_image, color_image):
        result_rgb = cv2.bitwise_and(color_image, color_image, mask=combined_mask)
        
        masked_depth = np.zeros_like(depth_image)
        masked_depth[combined_mask == 255] = depth_image[combined_mask == 255]
        result_depth = masked_depth

        o3d_color = o3d.geometry.Image(result_rgb.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(result_depth.astype(np.uint16))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

        return result_rgb, result_depth, rgbd_image
    
    def preprocess_scanned_cloud(self, pcd):
        """Preprocess scanned point cloud for registration"""
        if len(pcd.points) == 0:
            return None, None
        
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        if len(pcd.points) == 0:
            return None, None
        
        # Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
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
        
        result = o3d.pipelines.registration.registration_icp(
            source, target, 
            max_correspondence_distance=distance_threshold,
            init=ransac_result.transformation, 
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result
    
    def register_point_clouds(self, scanned_cloud):
        """Main registration pipeline"""
        try:
            # Preprocess scanned cloud
            scanned_down, scanned_fpfh = self.preprocess_scanned_cloud(scanned_cloud)
            
            if scanned_down is None or scanned_fpfh is None:
                self.get_logger().warn("Failed to preprocess scanned point cloud")
                return None, False
            
            # Global registration
            result_ransac = self.perform_global_registration(
                scanned_down, scanned_fpfh, 
                self.canonical_model, self.canonical_model.fpfh
            )
            
            self.get_logger().info(f"Global registration fitness: {result_ransac.fitness:.4f}")
            
            if result_ransac.fitness < 0.1:  # Very low fitness, likely failed
                self.get_logger().warn("Global registration failed - low fitness")
                return None, False
            
            # Refine with ICP
            result_icp = self.refine_registration(scanned_cloud, self.canonical_model, result_ransac)
            
            self.get_logger().info(f"ICP registration fitness: {result_icp.fitness:.4f}")
            
            # Check if registration was successful
            success = result_icp.fitness > self.registration_threshold
            
            if success:
                self.get_logger().info("Registration successful!")
                self.last_transformation = result_icp.transformation
                self.registration_success = True
                return result_icp.transformation, True
            else:
                self.get_logger().warn(f"Registration failed - fitness {result_icp.fitness:.4f} below threshold {self.registration_threshold}")
                return None, False
                
        except Exception as e:
            self.get_logger().error(f"Registration error: {str(e)}")
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
    
    def convert_o3d_to_ros2_pointcloud2(self, o3d_cloud, frame_id="map", timestamp=None):
        """Convert Open3D point cloud to ROS2 PointCloud2 message"""
        header = Header()
        header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
        header.frame_id = frame_id

        points = np.asarray(o3d_cloud.points)
        has_colors = o3d_cloud.has_colors()
        colors = np.asarray(o3d_cloud.colors) if has_colors else None

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]

        if has_colors:
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))

        point_step = 16 if has_colors else 12
        data = bytearray()

        for i in range(len(points)):
            x, y, z = points[i]
            data.extend(struct.pack('fff', x, y, z))

            if has_colors:
                r, g, b = (colors[i] * 255).astype(np.uint8)
                rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
                data.extend(struct.pack('f', rgb))

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = point_step
        pointcloud_msg.row_step = point_step * len(points)
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = data

        return pointcloud_msg
    
    def process_and_register(self):
        """Main processing pipeline with registration"""
        if self.color_image is None or self.depth_image is None or self.canonical_model is None:
            return

        try:
            # Generate point cloud from RGBD (same as original code)
            depth_resized = cv2.resize(self.depth_image, (self.color_image.shape[1], self.color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color_mask = self.create_color_mask(self.color_image)
            depth_mask = self.create_depth_mask(depth_resized, self.color_image.shape[:2])
            combined_mask = self.combine_masks(depth_mask, color_mask)
            clean_combined_mask = self.clean_mask_with_connected_components(combined_mask)
            
            filtered_color, filtered_depth, rgbd_image = self.generate_rgbd_image(
                clean_combined_mask, depth_resized, self.color_image)

            # Create point cloud from RGBD
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(
                width=self.color_image.shape[1], height=self.color_image.shape[0],
                fx=607.7908935546875, fy=607.75390625, cx=640.822509765625, cy=369.03350830078125)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
            
            # Translate point cloud to have its centroid at the origin
            centroid = pcd.get_center()
            pcd.translate(-centroid)

            # Publish original scanned point cloud
            point_cloud_msg = self.convert_o3d_to_ros2_pointcloud2(pcd)
            if point_cloud_msg:
                self.point_cloud_pub.publish(point_cloud_msg)
                self.get_logger().info("Published scanned point cloud")

            # Perform pose registration
            if len(pcd.points) > 100:  # Only attempt registration if we have sufficient points
                transformation, success = self.register_point_clouds(pcd)
                
                if success and transformation is not None:
                    # Apply transformation to create registered point cloud
                    registered_pcd = copy.deepcopy(pcd)
                    registered_pcd.transform(transformation)
                    
                    # Publish registered point cloud
                    registered_msg = self.convert_o3d_to_ros2_pointcloud2(registered_pcd, frame_id="canonical_frame")
                    if registered_msg:
                        self.registered_cloud_pub.publish(registered_msg)
                        self.get_logger().info("Published registered point cloud")
                    
                    # Log pose information
                    pose_msg = self.transformation_to_pose_msg(transformation)
                    self.get_logger().info(f"Estimated pose - Translation: [{pose_msg.translation.x:.3f}, {pose_msg.translation.y:.3f}, {pose_msg.translation.z:.3f}]")
                    self.get_logger().info(f"Estimated pose - Rotation: [{pose_msg.rotation.x:.3f}, {pose_msg.rotation.y:.3f}, {pose_msg.rotation.z:.3f}, {pose_msg.rotation.w:.3f}]")
                else:
                    self.get_logger().warn("Registration failed - no pose estimate available")
            else:
                self.get_logger().warn(f"Insufficient points for registration: {len(pcd.points)}")

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in processing and registration: {str(e)}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # Path to your canonical model - modify this path
    canonical_model_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"
    
    # You can also pass this as a command line argument
    import sys
    if len(sys.argv) > 1:
        canonical_model_path = sys.argv[1]
    
    try:
        node = PoseRegistrationNode(canonical_model_path)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to start node: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()