from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Scanned Target Point Cloud Processing Node
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='inteld435i_target_scanned_point_cloud_node',
            name='inteld435i_target_scanned_point_cloud_node'
        ),

        # Canonical Source Point Cloud Publisher
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='inteld435i_source_canonical_point_cloud_node',
            name='inteld435i_source_canonical_point_cloud_node'
        ),

        # Alignment and Scaling Node
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='inteld435i_align_and_scale_point_clouds',
            name='inteld435i_align_and_scale_point_clouds',
        ),

        # Ṕose Registration Node
        Node(
            package='xarm6_3d_detection_and_tracking',
            executable='inteld435i_register_pose_point_clouds',
            name='inteld435i_register_pose_point_clouds',
        )
    ])