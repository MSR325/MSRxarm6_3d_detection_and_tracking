from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Azure Kinect ROS Driver Node
        Node(
            package='azure_kinect_ros_driver',
            executable='node',
            name='azure_kinect',
            parameters=[
                {'rgb_camera_info': True},
                {'point_cloud': True},           # Enable point cloud publishing
                {'rgb_point_cloud': True},       # Enable RGB point cloud
                {'point_cloud_in_depth_frame': True}  # Publish in depth frame
            ]
        ),
    ])