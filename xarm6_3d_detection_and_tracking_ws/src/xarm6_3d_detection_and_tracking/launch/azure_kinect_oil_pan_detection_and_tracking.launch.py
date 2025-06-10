from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    canonical_model_path_arg = DeclareLaunchArgument(
        'canonical_model_path',
        default_value='/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply',
        description='Path to the canonical oil pan model (.ply file)'
    )
    
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='1.0',
        description='Voxel size for point cloud downsampling in registration'
    )
    
    registration_frequency_arg = DeclareLaunchArgument(
        'registration_frequency',
        default_value='2.0',
        description='Frequency (Hz) for performing registration updates'
    )
    
    min_component_area_arg = DeclareLaunchArgument(
        'min_component_area',
        default_value='1000',
        description='Minimum component area for mask cleaning'
    )
    
    color_resolution_arg = DeclareLaunchArgument(
        'color_resolution',
        default_value='720P',
        description='Azure Kinect color resolution (720P, 1080P, 2160P)'
    )
    
    depth_mode_arg = DeclareLaunchArgument(
        'depth_mode',
        default_value='NFOV_UNBINNED',
        description='Azure Kinect depth mode (NFOV_UNBINNED, WFOV_2X2BINNED, etc.)'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Azure Kinect frame rate'
    )

    # Add the use_rviz argument
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to launch RViz2 for visualization'
    )

    return LaunchDescription([
        # Launch arguments
        canonical_model_path_arg,
        voxel_size_arg,
        registration_frequency_arg,
        min_component_area_arg,
        color_resolution_arg,
        depth_mode_arg,
        fps_arg,
        use_rviz_arg,
        
        # Azure Kinect ROS Driver Node
        Node(
            package='azure_kinect_ros_driver',
            executable='node',
            name='azure_kinect',
            namespace='azure_kinect',
            parameters=[{
                'color_enabled': True,
                'depth_enabled': True,
                'color_resolution': LaunchConfiguration('color_resolution'),
                'depth_mode': LaunchConfiguration('depth_mode'),
                'fps': LaunchConfiguration('fps'),
                'rgb_camera_info': True,
                'point_cloud': True,
                'rgb_point_cloud': True,
                'point_cloud_in_depth_frame': True,
                'tf_prefix': 'azure_kinect_',
                'recording_file': '',
                'recording_loop_enabled': False,
                'body_tracking_enabled': False,
                'body_tracking_smoothing_factor': 0.0,
                'rescale_ir_to_mono8': False,
                'ir_mono8_scaling_factor': 1.0,
                'imu_rate_target': 1600,
                'wired_sync_mode': 0,  # Standalone mode
                'subordinate_delay_off_master_usec': 0,
                'depth_delay_off_color_usec': 0
            }],
            remappings=[
                # Remap to standard topic names for compatibility
                ('rgb/image_raw', '/rgb/image_raw'),
                ('depth/image_raw', '/depth/image_raw'),
                ('rgb/camera_info', '/rgb/camera_info'),
                ('depth/camera_info', '/depth/camera_info'),
                ('points2', '/azure_kinect/points2'),
                ('imu', '/azure_kinect/imu')
            ],
            output='screen'
        ),
        
        # Azure Kinect Source Scanned Point Cloud Node
        Node(
            package='xarm6_3d_detection_and_tracking',  # Replace with your actual package name
            executable='azure_kinect_source_scanned_point_cloud',  # Replace with your actual executable name
            name='azure_kinect_source_scanned_point_cloud',
            parameters=[{
                'min_component_area': LaunchConfiguration('min_component_area'),
                # Add any other parameters your node needs
                'depth_min': 250,
                'depth_max': 800,
                'color_lower_hsv': [0, 0, 0],
                'color_upper_hsv': [180, 255, 70]
            }],
            remappings=[
                # Ensure the node subscribes to the correct topics
                ('/rgb/image_raw', '/rgb/image_raw'),
                ('/depth/image_raw', '/depth/image_raw')
            ],
            output='screen'
        ),
        
        # Real-time Registration Node
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='perform_pose_reg_on_source_scanned',
            name='perform_pose_reg_on_source_scanned',
            parameters=[{
                'canonical_model_path': LaunchConfiguration('canonical_model_path'),
                'voxel_size': LaunchConfiguration('voxel_size'),
                'registration_frequency': LaunchConfiguration('registration_frequency'),
                'min_component_area': LaunchConfiguration('min_component_area')
            }],
            remappings=[
                # Subscribe to the filtered point cloud from the scanning node
                ('/filtered_source_scanned_point_cloud', '/filtered_source_scanned_point_cloud'),
                # Or if using RGB-D images directly:
                ('/rgb/image_raw', '/rgb/image_raw'),
                ('/depth/image_raw', '/depth/image_raw')
            ],
            output='screen'
        ),
        
        # Optional: Static transform publisher for sensor mounting
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='azure_kinect_base_link',
            arguments=[
                '0', '0', '0',  # x, y, z translation
                '0', '0', '0', '1',  # quaternion rotation (x, y, z, w)
                'base_link',  # parent frame
                'azure_kinect_link'  # child frame
            ]
        ),
        
        # Optional: RViz for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(
                get_package_share_directory('xarm6_3d_detection_and_tracking'),  # Replace with your package
                'config',
                'oil_pan_detection_and_tracking.rviz'  # Create this config file
            )],
            condition=IfCondition(LaunchConfiguration('use_rviz'))  # Fixed: Use IfCondition
        )
    ])