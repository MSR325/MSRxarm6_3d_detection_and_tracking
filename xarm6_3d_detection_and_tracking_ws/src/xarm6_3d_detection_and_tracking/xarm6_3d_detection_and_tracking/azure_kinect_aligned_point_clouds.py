import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
import struct
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudAlignmentNode(Node):
    def __init__(self):
        super().__init__('azure_kinect_aligned_point_clouds')

        # Subscriptions
        self.create_subscription(PointCloud2, '/filtered_target_point_cloud', self.source_callback, 10)
        self.create_subscription(PointCloud2, '/filtered_source_point_cloud', self.target_callback, 10)

        # Publishers
        self.aligned_source_pub = self.create_publisher(PointCloud2, '/aligned_source_cloud', 10)
        self.target_pub = self.create_publisher(PointCloud2, '/registered_target_cloud', 10)

        # Storage
        self.source_cloud = None
        self.target_cloud = None

    def source_callback(self, msg):
        self.source_cloud = self.convert_ros2_to_o3d(msg)
        self.source_header = msg.header
        self.get_logger().info('Received source point cloud from /filtered_target_point_cloud')
        self.try_align()

    def target_callback(self, msg):
        self.target_cloud = self.convert_ros2_to_o3d(msg)
        self.target_header = msg.header
        self.get_logger().info('Received target point cloud from /filtered_source_point_cloud')
        self.try_align()

    def convert_ros2_to_o3d(self, msg):
        points = []
        for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")):
            points.append([p[0], p[1], p[2]])

        if not points:
            return None

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        return cloud

    def convert_o3d_to_ros2(self, cloud, header):
        points = np.asarray(cloud.points)
        if len(points) == 0:
            return None

        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1)
        ]

        cloud_data = [tuple(p) for p in points]

        msg = pc2.create_cloud(header, fields, cloud_data)
        return msg

    def try_align(self):
        if self.source_cloud is not None and self.target_cloud is not None:
            self.get_logger().info("Both point clouds received — performing alignment")

            # Downsample
            voxel_size = 0.005
            source_down = self.source_cloud.voxel_down_sample(voxel_size)
            target_down = self.target_cloud.voxel_down_sample(voxel_size)

            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

            # FPFH feature extraction
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100))

            # Global RANSAC alignment
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                max_correspondence_distance=0.03,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.03)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000))

            self.get_logger().info(f"RANSAC Fitness: {result_ransac.fitness:.4f}, Inlier RMSE: {result_ransac.inlier_rmse:.4f}")

            # Local ICP refinement
            result_icp = o3d.pipelines.registration.registration_icp(
                self.source_cloud, self.target_cloud, 0.02,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())

            self.get_logger().info(f"ICP Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")

            # Transform the source cloud
            aligned_source = self.source_cloud.transform(result_icp.transformation)

            # Publish both processed clouds
            aligned_source_msg = self.convert_o3d_to_ros2(self.source_cloud, self.source_header)
            target_msg = self.convert_o3d_to_ros2(self.target_cloud, self.target_header)

            if aligned_source_msg:
                self.aligned_source_pub.publish(aligned_source_msg)
                self.get_logger().info('Published aligned source cloud to /aligned_source_cloud')

            if target_msg:
                self.target_pub.publish(target_msg)
                self.get_logger().info('Published target cloud to /registered_target_cloud')

            # Optional visualization
            self.source_cloud.paint_uniform_color([1, 0, 0])
            self.target_cloud.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([self.source_cloud, self.target_cloud])

            # Reset after alignment
            self.source_cloud = None
            self.target_cloud = None

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudAlignmentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()