from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Azure Kinect ROS Driver Node
        Node(
            package='azure_kinect_ros_driver',
            executable='node',
            name='azure_kinect',
            parameters=[{'rgb_camera_info': True}]
        ),
        
        # XArm6 3D Detection and Tracking Node
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='azure_kinect_oil_pan_tracking_node',
            name='azure_kinect_oil_pan_tracking_node',
            parameters=[
                {'min_component_area': 2500},
                {'depth_min': 300},
                {'depth_max': 800}
            ]
        )
    ])