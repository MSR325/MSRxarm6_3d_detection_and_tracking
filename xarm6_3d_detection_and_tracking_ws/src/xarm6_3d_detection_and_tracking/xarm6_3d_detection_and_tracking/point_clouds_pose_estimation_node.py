import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from std_srvs.srv import Empty

class PointCloudsPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('point_clouds_pose_estimation_node')
        self.bridge = CvBridge()
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        
        self.color_image = None
        self.depth_image = None
        self.min_component_area = 1000
        self.camera_intrinsics = self.setup_camera_intrinsics()
        self.vis = None
        self.point_cloud = None
        self.setup_visualizer()

        # Service to save current point cloud as model reference
        self.create_service(Empty, 'save_model_point_cloud', self.save_model_service_callback)

        # Load model for alignment
        self.model_pcd = o3d.io.read_point_cloud("model_object.ply")  # path to your model point cloud

    def setup_camera_intrinsics(self):
        width, height = 1280, 720
        fx, fy = 607.79, 607.75
        cx, cy = 640.82, 369.03
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        self.get_logger().info(f'Camera intrinsics: {width}x{height}, fx={fx}, fy={fy}, cx={cx}, cy={cy}')
        return intrinsics

    def setup_visualizer(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Point Cloud Visualization', width=800, height=600)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(frame)
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_display()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_display()

    def create_color_mask(self, color_image):
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lo_black, up_black = np.array([0, 0, 0]), np.array([180, 255, 70])
        return cv2.inRange(hsv_img, lo_black, up_black)

    def create_depth_mask(self, depth_image):
        return cv2.inRange(depth_image, 350, 525)

    def combine_masks(self, depth_mask, color_mask):
        return cv2.bitwise_and(depth_mask, color_mask)

    def clean_mask_with_connected_components(self, combined_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        clean_mask = np.zeros_like(combined_mask)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_idx = np.argmax(areas) + 1
            if stats[largest_component_idx, cv2.CC_STAT_AREA] >= self.min_component_area:
                clean_mask[labels == largest_component_idx] = 255
        return clean_mask

    def generate_rgbd_image(self, combined_mask, depth_image, color_image):
        result_rgb = cv2.bitwise_and(color_image, color_image, mask=combined_mask)
        result_depth = cv2.bitwise_and(depth_image, depth_image, mask=combined_mask)
        result_rgb_o3d = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(result_rgb_o3d.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(result_depth.astype(np.uint16))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

    def create_point_cloud_from_color_and_depth(self, rgbd_image):
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd

    def generate_center_line(self, pcd):
        centroid = pcd.get_center()
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return None
        max_z_idx = np.argmax(points[:, 2])
        feature_point = points[max_z_idx]
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([centroid, feature_point]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        return line, centroid, feature_point

    def align_and_register_model(self, source_pcd, target_pcd):
        voxel_size = 0.005
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)

        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100))

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=source_down, target=target_down, source_feature=source_fpfh, target_feature=target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=0.015,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.015)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
        )

        # Estimate normals for full-res clouds before ICP
        source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.01,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result_icp.transformation

    def update_point_cloud_visualization(self, pcd, line=None, model=None):
        if len(pcd.points) == 0:
            self.get_logger().warn('Generated point cloud is empty')
            return

        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)
        if line:
            self.vis.add_geometry(line)
        if model:
            self.vis.add_geometry(model)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        self.vis.poll_events()
        self.vis.update_renderer()

    def process_and_display(self):
        if self.color_image is None or self.depth_image is None:
            return
        try:
            color_mask = self.create_color_mask(self.color_image)
            depth_mask = self.create_depth_mask(self.depth_image)
            combined_mask = self.combine_masks(depth_mask, color_mask)
            clean_mask = self.clean_mask_with_connected_components(combined_mask)
            rgbd_image = self.generate_rgbd_image(clean_mask, self.depth_image, self.color_image)
            pcd = self.create_point_cloud_from_color_and_depth(rgbd_image)


            if len(pcd.points) == 0:
                self.get_logger().warn('Generated point cloud is empty — skipping.')
                return
            
            self.latest_pcd = pcd # Save latest point cloud for capture service

            line, centroid, feature_point = self.generate_center_line(pcd)
            transformation = self.align_and_register_model(self.model_pcd, pcd)
            self.model_pcd.transform(transformation)

            self.update_point_cloud_visualization(pcd, line, self.model_pcd)

        except Exception as e:
            self.get_logger().error(f'Processing error: {str(e)}')

    def save_model_service_callback(self, request, response):
        if self.latest_pcd is not None and len(self.latest_pcd.points) > 0:
            o3d.io.write_point_cloud("model_object.ply", self.latest_pcd)
            self.get_logger().info('Current point cloud saved as model_object.ply')
        else:
            self.get_logger().warn('No valid point cloud available to save.')
        return response

    def destroy_node(self):
        if self.vis:
            self.vis.destroy_window()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudsPoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()