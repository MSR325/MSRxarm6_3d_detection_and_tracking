from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Azure Kinect ROS Driver Node
        Node(
            package='azure_kinect_ros_driver',
            executable='node',
            name='azure_kinect',
            parameters=[{
                'color_enabled': True,
                'depth_enabled': True,
                'color_resolution': '720P',
                'depth_mode': 'NFOV_UNBINNED',
                'fps': 30,
                'rgb_camera_info': True,
                'point_cloud': True,
                'rgb_point_cloud': True,
                'point_cloud_in_depth_frame': True
            }]
        ),

        # Scalable Transform Publisher Node
        # Transforms from "map" to "camera_base" frame
        Node(
            package='xarm6_3d_detection_and_tracking',  # Replace with your actual package name
            executable='scalable_transform_publisher',
            name='map_to_camera_transform',
            parameters=[{
                'parent_frame': 'map',
                'child_frame': 'camera_base',
                'publish_rate': 30.0,  # Match the camera fps for smooth visualization
                
                # Initial transform parameters (adjust as needed)
                'translation_x': 0.0,
                'translation_y': 0.0,
                'translation_z': 0.0,
                
                'rotation_x': 0.0,  # Roll in degrees
                'rotation_y': 0.0,  # Pitch in degrees
                'rotation_z': 0.0,  # Yaw in degrees
                
                'scale_x': 0.0001,
                'scale_y': 0.0001,
                'scale_z': 0.0001,
                
                # Animation parameters (optional)
                'animate_rotation': False,
                'rotation_speed': 1.0,
                'rotation_axis': 'z'
            }]
        ),

        # Scanned Point Cloud Processing Node
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='azure_kinect_source_scanned_point_cloud',
            name='azure_kinect_source_scanned_point_cloud',
            parameters=[{
                'target_frame': 'camera_base'  # Changed to camera_base to align with canonical model
            }]
        ),

        # Canonical Model Publisher
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='azure_kinect_target_canonical_point_cloud',
            name='azure_kinect_target_canonical_point_cloud',
            parameters=[{
                'model_path': '/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply',
                'publish_frame': 'camera_base'  # Changed to camera_base to align with scanned point cloud
            }]
        ),

        # Alignment and Scaling Node
        # Node(
        #     package='xarm6_3d_detection_and_tracking',
        #     executable='azure_kinect_align_and_scale',
        #     name='azure_kinect_align_and_scale',
        #     parameters=[{
        #         'aligned_frame': 'camera_base'  # Changed to camera_base for consistency
        #     }]
        # )
    ])