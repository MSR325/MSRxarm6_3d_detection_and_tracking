#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
import os

class AlignAndScaleNode(Node):
    def __init__(self):
        super().__init__('align_and_scale_node')

        self.declare_parameter('model_path', 'model.ply')
        self.declare_parameter('scan_path', 'scan.ply')
        self.declare_parameter('output_path', 'scaled_model.ply')
        self.declare_parameter('visualize', True)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        scan_path  = self.get_parameter('scan_path').get_parameter_value().string_value
        output_path = self.get_parameter('output_path').get_parameter_value().string_value
        visualize = self.get_parameter('visualize').get_parameter_value().bool_value

        if not os.path.exists(model_path) or not os.path.exists(scan_path):
            self.get_logger().error("Model or scan file path does not exist.")
            return

        source = o3d.io.read_point_cloud(model_path)
        target = o3d.io.read_point_cloud(scan_path)

        aligned_scaled_model = self.align_and_scale(source, target)

        o3d.io.write_point_cloud(output_path, aligned_scaled_model)
        self.get_logger().info(f"Aligned and scaled model saved to {output_path}")

        if visualize:
            target.paint_uniform_color([1, 0, 0])  # Red
            aligned_scaled_model.paint_uniform_color([0, 1, 0])  # Green
            o3d.visualization.draw_geometries([target, aligned_scaled_model])

    def compute_centroid(self, pcd):
        return np.mean(np.asarray(pcd.points), axis=0)

    def identify_peduncle_point_model(self, pcd):
        z_coords = np.asarray(pcd.points)[:, 2]
        threshold = np.percentile(z_coords, 98)
        top_points = np.asarray(pcd.points)[z_coords > threshold]
        return np.mean(top_points, axis=0)

    def identify_peduncle_point_scan(self, pcd):
        y_coords = np.asarray(pcd.points)[:, 1]
        threshold = np.percentile(y_coords, 98)
        top_points = np.asarray(pcd.points)[y_coords > threshold]
        return np.mean(top_points, axis=0)

    def compute_rotation(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return np.eye(3)  # No rotation needed
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return self.axis_angle_to_rotation_matrix(axis, angle)

    def axis_angle_to_rotation_matrix(self, axis, angle):
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def apply_rotation(self, pcd, R):
        pcd.rotate(R, center=(0, 0, 0))
        return pcd

    def get_dimensions(self, pcd):
        bbox = pcd.get_axis_aligned_bounding_box()
        return bbox.get_extent()

    def scale_point_cloud(self, source_pcd, target_dims, source_dims=None):
        if source_dims is None:
            source_dims = self.get_dimensions(source_pcd)
        scale_factors = target_dims / source_dims
        scaled_points = np.asarray(source_pcd.points) * scale_factors
        source_pcd.points = o3d.utility.Vector3dVector(scaled_points)
        return source_pcd

    def align_and_scale(self, source, target):
        # Compute centroids
        centroid_source = self.compute_centroid(source)
        centroid_target = self.compute_centroid(target)

        # Translate both to origin
        source.translate(-centroid_source)
        target.translate(-centroid_target)

        # Compute orientation lines (vector from centroid to peduncle point)
        feature_source = self.identify_peduncle_point_model(source)
        feature_target = self.identify_peduncle_point_scan(target)

        vec_source = feature_source
        vec_target = feature_target

        # Align source to target orientation
        R = self.compute_rotation(vec_source, vec_target)
        self.apply_rotation(source, R)

        # Compute scaling
        target_dims = self.get_dimensions(target)
        source_dims = self.get_dimensions(source)
        scaled_source = self.scale_point_cloud(source, target_dims, source_dims)

        return scaled_source

def main(args=None):
    rclpy.init(args=args)
    node = AlignAndScaleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
