import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Transform, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import struct
import copy
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import threading
import time

class RealTimeRegistrationNode(Node):
    def __init__(self):
        super().__init__('perform_pose_reg_on_source_scanned')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('canonical_model_path', '/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_front_pc_10000.ply')
        self.declare_parameter('voxel_size', 0.003) # 5 mm Also experiment with 0.003 or 0.010
        self.declare_parameter('min_component_area', 1000)
        self.declare_parameter('registration_frequency', 2.0)  # Hz
        
        self.canonical_model_path = self.get_parameter('canonical_model_path').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.min_component_area = self.get_parameter('min_component_area').value
        self.registration_freq = self.get_parameter('registration_frequency').value
        
        # Subscribers
        self.source_pc_sub = self.create_subscription(
            PointCloud2, '/filtered_source_scanned_point_cloud', self.source_pc_callback, 10)

        # Publishers
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/filtered_source_scanned_point_cloud', 10)
        self.registered_cloud_pub = self.create_publisher(
            PointCloud2, '/registered_point_cloud', 10)
        self.canonical_cloud_pub = self.create_publisher(
            PointCloud2, '/canonical_point_cloud', 10)
        
        # TF broadcaster for pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Image and point cloud storage
        self.color_image = None
        self.depth_image = None
        self.current_scan_pcd = None
        self.canonical_pcd = None
        self.canonical_pcd_down = None
        self.canonical_fpfh = None
        
        # Registration state
        self.last_transformation = np.identity(4)
        self.registration_lock = threading.Lock()
        
        # Load canonical model
        self.load_canonical_model()
        
        # Registration timer
        self.registration_timer = self.create_timer(
            1.0 / self.registration_freq, self.perform_registration)
        
        self.get_logger().info("Real-time registration node initialized")
    
    def load_canonical_model(self):
        """Load and preprocess the canonical model"""
        try:
            self.canonical_pcd = o3d.io.read_point_cloud(self.canonical_model_path)
            if len(self.canonical_pcd.points) == 0:
                self.get_logger().error(f"Failed to load canonical model from {self.canonical_model_path}")
                return
            
            # Preprocess canonical model
            self.canonical_pcd_down, self.canonical_fpfh = self.preprocess_point_cloud(
                self.canonical_pcd, self.voxel_size)
            
            # Publish canonical model for visualization
            canonical_msg = self.convert_o3d_to_ros2_pointcloud2(
                self.canonical_pcd, frame_id="canonical_frame")
            self.canonical_cloud_pub.publish(canonical_msg)
            
            self.get_logger().info(f"Loaded canonical model with {len(self.canonical_pcd.points)} points")
            
        except Exception as e:
            self.get_logger().error(f"Error loading canonical model: {str(e)}")
    
    def preprocess_point_cloud(self, pcd, voxel_size):
        """Preprocess point cloud for registration"""
        pcd_down = pcd.voxel_down_sample(voxel_size)
        print(f"[DEBUG] Points after downsampling: {len(pcd_down.points)}")
        
        if len(pcd_down.points) < 3000:
            print(f"[WARN] Too few points ({len(pcd_down.points)}) after downsampling. Try lowering voxel_size.")
            return None, None

        radius_normal = voxel_size * 3 # Originally 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 10 # Originally 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        print(f"[DEBUG] FPFH features shape: {pcd_fpfh.data.shape}")

        return pcd_down, pcd_fpfh
    
    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        """Perform RANSAC-based global registration"""
        distance_threshold = voxel_size * 1.5
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        return result
    
    def refine_registration(self, source, target, initial_transformation, voxel_size):
        """Refine registration using ICP"""
        distance_threshold = voxel_size * 0.4
        
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        return result
    
    def perform_registration(self):
        """Perform registration between current scan and canonical model"""
        if self.current_scan_pcd is None or self.canonical_pcd is None:
            return
        
        with self.registration_lock:
            try:
                # Preprocess current scan
                scan_down, scan_fpfh = self.preprocess_point_cloud(
                    self.current_scan_pcd, self.voxel_size)
                
                if len(scan_down.points) < 100:  # Minimum points threshold
                    self.get_logger().warn(f"Insufficient points {(scan_down.points)} in scan for registration")
                    return
                
                # Global registration
                result_ransac = self.execute_global_registration(
                    scan_down, self.canonical_pcd_down, 
                    scan_fpfh, self.canonical_fpfh, self.voxel_size)
                
                if result_ransac.fitness < 0.1:  # Low fitness threshold
                    self.get_logger().warn(f"Poor global registration fitness: {result_ransac.fitness}")
                
                # Compute normals for ICP refinement
                radius_normal = self.voxel_size * 2
                self.current_scan_pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
                self.canonical_pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
                
                # Refine with ICP
                result_icp = self.refine_registration(
                    self.current_scan_pcd, self.canonical_pcd, 
                    result_ransac.transformation, self.voxel_size)
                
                # Update transformation
                self.last_transformation = result_icp.transformation
                
                # Create registered point cloud for visualization
                registered_pcd = copy.deepcopy(self.current_scan_pcd)
                registered_pcd.transform(self.last_transformation)
                
                # Publish registered point cloud
                registered_msg = self.convert_o3d_to_ros2_pointcloud2(
                    registered_pcd, frame_id="canonical_frame")
                self.registered_cloud_pub.publish(registered_msg)
                
                # Publish transform
                self.publish_transform(self.last_transformation)
                
                self.get_logger().info(
                    f"Registration - RANSAC fitness: {result_ransac.fitness:.3f}, "
                    f"ICP fitness: {result_icp.fitness:.3f}")
                
            except Exception as e:
                self.get_logger().error(f"Registration error: {str(e)}")
    
    def publish_transform(self, transformation):
        """Publish the transformation as a TF transform"""
        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = "canonical_frame"
        transform_msg.child_frame_id = "scanned_object_frame"
        
        # Extract translation
        transform_msg.transform.translation.x = transformation[0, 3]
        transform_msg.transform.translation.y = transformation[1, 3]
        transform_msg.transform.translation.z = transformation[2, 3]
        
        # Extract rotation (convert rotation matrix to quaternion)
        rotation_matrix = transformation[:3, :3]
        # Simple conversion - for production use scipy.spatial.transform.Rotation
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s
        
        transform_msg.transform.rotation.x = x
        transform_msg.transform.rotation.y = y
        transform_msg.transform.rotation.z = z
        transform_msg.transform.rotation.w = w
        
        self.tf_broadcaster.sendTransform(transform_msg)
    
    def convert_o3d_to_ros2_pointcloud2(self, o3d_cloud, frame_id="map", timestamp=None):
        """Convert Open3D point cloud to ROS2 PointCloud2 message"""
        header = Header()
        header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
        header.frame_id = frame_id

        points = np.asarray(o3d_cloud.points)
        has_colors = o3d_cloud.has_colors()
        colors = np.asarray(o3d_cloud.colors) if has_colors else None

        if len(points) == 0:
            return None

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
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
    
    def convert_ros2_to_o3d_pointcloud(self, ros_pc):
        points = []
        for point in pc2.read_points(ros_pc, field_names=("x", "y", "z"), skip_nans=True):
            # Unpack individual float values
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            points.append([x, y, z])

        if len(points) == 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        #o3d.visualization.draw_geometries([pcd])

        return pcd
    
    def source_pc_callback(self, msg):
        """Callback for receiving the scanned source point cloud"""
        try:
            pcd = self.convert_ros2_to_o3d_pointcloud(msg)
            if pcd is not None:
                with self.registration_lock:
                    self.current_scan_pcd = pcd
                    self.get_logger().info(f"Source scanned point cloud has {len(self.current_scan_pcd.points)} points.")

        except Exception as e:
            self.get_logger().error(f"Error converting incoming point cloud: {str(e)}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeRegistrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()