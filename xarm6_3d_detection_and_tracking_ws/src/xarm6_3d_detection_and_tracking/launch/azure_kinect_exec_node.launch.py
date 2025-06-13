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
                'color_resolution': '720P',      # Options: '720P', '1080P', '2160P', etc.
                'depth_mode': 'NFOV_UNBINNED',   # Options: 'NFOV_UNBINNED', 'WFOV_2X2BINNED', etc.
                'fps': 30, # 30
                'rgb_camera_info': True,
                'point_cloud': True,
                'rgb_point_cloud': True,
                'point_cloud_in_depth_frame': True             # Set to True if you also want point clouds published
            }]
        ),
    ])