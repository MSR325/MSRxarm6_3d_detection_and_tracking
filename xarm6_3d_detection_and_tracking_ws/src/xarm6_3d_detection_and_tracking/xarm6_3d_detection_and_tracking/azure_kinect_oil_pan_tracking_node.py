import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from std_srvs.srv import Empty
import os
import threading
import time

class AzureKinectOilPanTrackingNode(Node):
    def __init__(self):
        super().__init__('azure_kinect_oil_pan_tracking_node')
        self.bridge = CvBridge()
        
        # Subscribers - Updated topic names for Azure Kinect
        self.color_sub = self.create_subscription(
            Image, '/rgb/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/depth_to_rgb/image_raw', self.depth_callback, 10)
        
        # Image storage
        self.color_image = None
        self.depth_image = None
        self.image_lock = threading.Lock()
        
        # Parameters for oil pan detection
        self.min_component_area = 2000  # Increased for oil pan
        self.depth_min = 300  # mm - closer objects
        self.depth_max = 800  # mm - further objects
        
        # Camera intrinsics for Azure Kinect
        self.camera_intrinsics = self.setup_camera_intrinsics()
        
        # Visualization
        self.vis = None
        self.point_cloud = None
        self.latest_pcd = None
        self.model_pcd = None
        self.setup_visualizer()
        
        # Service to save current point cloud as model reference
        self.create_service(Empty, 'save_model_point_cloud', self.save_model_service_callback)
        
        # Load or initialize model
        self.load_model()
        
        # Tracking state
        self.tracking_initialized = False
        self.last_transformation = np.eye(4)
        
        self.get_logger().info('Oil Pan Tracking Node initialized')

    def setup_camera_intrinsics(self):
        # Azure Kinect DK camera intrinsics (adjust these based on your calibration)
        width, height = 1280, 720
        fx, fy = 607.79, 607.75
        cx, cy = 640.82, 369.03
        
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        self.get_logger().info(f'Camera intrinsics: {width}x{height}, fx={fx}, fy={fy}, cx={cx}, cy={cy}')
        return intrinsics

    def setup_visualizer(self):
        """Setup Open3D visualizer"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Oil Pan Tracking', width=1200, height=800)
        
        # Add coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(frame)
        
        # Initialize point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)

    def load_model(self):
        """Load or create model point cloud"""
        model_path = "oil_pan_model.ply"
        if os.path.exists(model_path):
            self.model_pcd = o3d.io.read_point_cloud(model_path)
            self.get_logger().info(f'Loaded model from {model_path}')
        else:
            self.get_logger().warn(f'Model file {model_path} not found. Please save a model first.')
            self.model_pcd = o3d.geometry.PointCloud()

    def color_callback(self, msg):
        """Callback for color image"""
        with self.image_lock:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_display()

    def depth_callback(self, msg):
        """Callback for depth image"""
        with self.image_lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_display()

    def create_color_mask(self, color_image):
        """Create color mask for oil pan detection"""
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Oil pan color range (dark metallic/black colors)
        # Adjust these HSV ranges based on your specific oil pan
        lo_range = np.array([0, 0, 0])      # Lower bound
        up_range = np.array([180, 255, 80])  # Upper bound (darker objects)
        
        color_mask = cv2.inRange(hsv_img, lo_range, up_range)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        return color_mask

    def create_depth_mask(self, depth_image):
        """Create depth mask to filter objects within reasonable distance"""
        return cv2.inRange(depth_image, self.depth_min, self.depth_max)

    def combine_masks(self, depth_mask, color_mask):
        """Combine depth and color masks"""
        return cv2.bitwise_and(depth_mask, color_mask)

    def clean_mask_with_connected_components(self, combined_mask):
        """Clean mask using connected component analysis with oil pan shape validation"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8)
        
        clean_mask = np.zeros_like(combined_mask)
        
        if num_labels > 1:
            # Get all components (excluding background)
            valid_components = []
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Oil pan shape validation criteria
                aspect_ratio = w / h if h > 0 else 0
                bounding_area = w * h
                fill_ratio = area / bounding_area if bounding_area > 0 else 0
                
                # Car oil pan characteristics:
                # - Elongated shape (length > width), aspect ratio typically 1.5-4.0
                # - Complex irregular shape, so fill ratio is lower than circular objects
                # - Reasonable size range for automotive oil pans
                is_valid_oil_pan = (
                    area >= self.min_component_area and  # Minimum size
                    area <= 50000 and  # Maximum reasonable size (adjust based on distance)
                    aspect_ratio >= 1.2 and  # Elongated shape (length > width)
                    aspect_ratio <= 5.0 and  # Not too elongated (avoid thin lines)
                    fill_ratio >= 0.3 and    # Not too sparse (avoid scattered noise)
                    fill_ratio <= 0.85 and   # Not too filled (oil pans have complex shapes)
                    w >= 50 and h >= 30      # Minimum reasonable dimensions in pixels
                )
                
                if is_valid_oil_pan:
                    valid_components.append((i, area, aspect_ratio, fill_ratio))
                    self.get_logger().debug(
                        f'Valid component {i}: area={area}, aspect_ratio={aspect_ratio:.2f}, fill_ratio={fill_ratio:.2f}')
            
            if valid_components:
                # Select the component with the best oil pan characteristics
                # Priority: larger area, reasonable aspect ratio (around 2-3), moderate fill ratio
                best_component = None
                best_score = 0
                
                for comp_idx, area, aspect_ratio, fill_ratio in valid_components:
                    # Scoring function for oil pan likelihood
                    # Prefer: moderate aspect ratio (2-3), moderate fill ratio (0.5-0.7), larger area
                    aspect_score = 1.0 - abs(aspect_ratio - 2.5) / 2.5  # Best at 2.5, decreases linearly
                    fill_score = 1.0 - abs(fill_ratio - 0.6) / 0.3      # Best at 0.6
                    area_score = min(area / 10000, 1.0)                 # Normalize area, cap at 1.0
                    
                    # Ensure scores are non-negative
                    aspect_score = max(0, aspect_score)
                    fill_score = max(0, fill_score)
                    
                    # Combined score
                    total_score = (aspect_score * 0.4 + fill_score * 0.3 + area_score * 0.3)
                    
                    self.get_logger().debug(
                        f'Component {comp_idx} scores: aspect={aspect_score:.2f}, fill={fill_score:.2f}, '
                        f'area={area_score:.2f}, total={total_score:.2f}')
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_component = comp_idx
                
                if best_component is not None:
                    clean_mask[labels == best_component] = 255
                    comp_info = next(c for c in valid_components if c[0] == best_component)
                    self.get_logger().info(
                        f'Selected oil pan component: area={comp_info[1]}, '
                        f'aspect_ratio={comp_info[2]:.2f}, fill_ratio={comp_info[3]:.2f}')
                else:
                    self.get_logger().warn('No valid oil pan component found based on shape analysis')
            else:
                self.get_logger().warn('No components passed oil pan shape validation')
                
        return clean_mask

    def generate_rgbd_image(self, mask, depth_image, color_image):
        """Generate RGBD image from masked color and depth"""
        # Apply mask to both images
        masked_color = cv2.bitwise_and(color_image, color_image, mask=mask)
        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        
        # Convert to Open3D format
        color_o3d = cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_o3d.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(masked_depth.astype(np.uint16))
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, 
            depth_scale=1000.0,  # Convert mm to meters
            depth_trunc=3.0,     # Truncate at 3 meters
            convert_rgb_to_intensity=False
        )
        
        return rgbd_image

    def create_point_cloud_from_rgbd(self, rgbd_image):
        """Create point cloud from RGBD image"""
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_intrinsics)
        
        # Transform to correct orientation (Azure Kinect coordinate system)
        transformation = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        pcd.transform(transformation)
        
        return pcd

    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud for better registration"""
        if len(pcd.points) == 0:
            return pcd
            
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample
        pcd = pcd.voxel_down_sample(0.005)  # 5mm voxel size
        
        # Estimate normals
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        return pcd

    def generate_characteristic_line(self, pcd):
        """Generate characteristic line for oil pan orientation"""
        if len(pcd.points) == 0:
            return None, None, None
            
        centroid = pcd.get_center()
        points = np.asarray(pcd.points)
        
        # For oil pan, find the furthest point in Y-axis (length direction)
        max_y_idx = np.argmax(points[:, 1])
        feature_point = points[max_y_idx]
        
        # Create line geometry
        line_points = [centroid, feature_point]
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red line
        
        return line, centroid, feature_point

    def align_and_register_model(self, source_pcd, target_pcd):
        """Align and register model to target using RANSAC + ICP"""
        if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
            return np.eye(4)
            
        # Preprocess both point clouds
        source_processed = self.preprocess_point_cloud(source_pcd.copy())
        target_processed = self.preprocess_point_cloud(target_pcd.copy())
        
        if len(source_processed.points) == 0 or len(target_processed.points) == 0:
            return np.eye(4)
        
        # Downsample for RANSAC
        voxel_size = 0.01
        source_down = source_processed.voxel_down_sample(voxel_size)
        target_down = target_processed.voxel_down_sample(voxel_size)
        
        # Compute FPFH features
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100)
        )
        
        # RANSAC registration
        distance_threshold = voxel_size * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        self.get_logger().info(
            f'RANSAC: Fitness={result_ransac.fitness:.4f}, RMSE={result_ransac.inlier_rmse:.4f}')
        
        # ICP refinement
        distance_threshold = 0.01  # 1cm
        result_icp = o3d.pipelines.registration.registration_icp(
            source_processed, target_processed,
            distance_threshold,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        self.get_logger().info(
            f'ICP: Fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}')
        
        return result_icp.transformation

    def update_visualization(self, pcd, line=None, model=None):
        """Update the visualization with current point cloud and model"""
        if len(pcd.points) == 0:
            self.get_logger().warn('Point cloud is empty - skipping visualization update')
            return
            
        # Clear previous geometries
        self.vis.clear_geometries()
        
        # Add current point cloud (green)
        pcd_vis = pcd.copy()
        pcd_vis.paint_uniform_color([0, 1, 0])  # Green
        self.vis.add_geometry(pcd_vis)
        
        # Add characteristic line if available
        if line is not None:
            self.vis.add_geometry(line)
        
        # Add aligned model if available (red)
        if model is not None and len(model.points) > 0:
            model_vis = model.copy()
            model_vis.paint_uniform_color([1, 0, 0])  # Red
            self.vis.add_geometry(model_vis)
        
        # Add coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(frame)
        
        # Update renderer
        self.vis.poll_events()
        self.vis.update_renderer()

    def process_and_display(self):
        """Main processing pipeline"""
        with self.image_lock:
            if self.color_image is None or self.depth_image is None:
                return
            
            color_img = self.color_image.copy()
            depth_img = self.depth_image.copy()
        
        try:
            # Step 1: Create masks
            color_mask = self.create_color_mask(color_img)
            depth_mask = self.create_depth_mask(depth_img)
            
            # Step 2: Combine masks
            combined_mask = self.combine_masks(depth_mask, color_mask)
            
            # Step 3: Clean mask with connected components
            clean_mask = self.clean_mask_with_connected_components(combined_mask)
            
            # Step 4: Generate RGBD image
            rgbd_image = self.generate_rgbd_image(clean_mask, depth_img, color_img)
            
            # Step 5: Create point cloud
            pcd = self.create_point_cloud_from_rgbd(rgbd_image)
            
            if len(pcd.points) == 0:
                self.get_logger().warn('Generated point cloud is empty - no oil pan detected')
                return
            
            # Step 6: Store latest point cloud
            self.latest_pcd = pcd.copy()
            
            # Step 7: Generate characteristic line
            line, centroid, feature_point = self.generate_characteristic_line(pcd)
            
            # Step 8: Align model if available
            aligned_model = None
            if len(self.model_pcd.points) > 0:
                transformation = self.align_and_register_model(self.model_pcd, pcd)
                aligned_model = self.model_pcd.copy()
                aligned_model.transform(transformation)
                self.last_transformation = transformation
                self.tracking_initialized = True
            
            # Step 9: Update visualization
            self.update_visualization(pcd, line, aligned_model)
            
            # Log detection info
            self.get_logger().info(f'Oil pan detected: {len(pcd.points)} points')
            
        except Exception as e:
            self.get_logger().error(f'Processing error: {str(e)}')

    def save_model_service_callback(self, request, response):
        """Service callback to save current point cloud as model"""
        if self.latest_pcd is not None and len(self.latest_pcd.points) > 0:
            model_path = "oil_pan_model.ply"
            o3d.io.write_point_cloud(model_path, self.latest_pcd)
            self.model_pcd = self.latest_pcd.copy()
            self.get_logger().info(f'Current point cloud saved as {model_path}')
        else:
            self.get_logger().warn('No valid point cloud available to save as model')
        
        return response

    def destroy_node(self):
        """Clean up resources"""
        if self.vis:
            self.vis.destroy_window()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AzureKinectOilPanTrackingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()