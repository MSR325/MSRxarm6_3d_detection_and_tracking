from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'xarm6_3d_detection_and_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.[yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chrisrvt',
    maintainer_email='christianvillarrealt@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inteld435i_listener_node = xarm6_3d_detection_and_tracking.inteld435i_listener_node:main',
            'inteld435i_filtered_listener_node = xarm6_3d_detection_and_tracking.inteld435i_filtered_listener_node:main',
            'inteld435i_rgbd_image_generator_node = xarm6_3d_detection_and_tracking.inteld435i_rgbd_image_generator_node:main',
            'inteld435i_point_clouds_pose_estimation_node = xarm6_3d_detection_and_tracking.inteld435i_point_clouds_pose_estimation_node:main',
            'inteld435i_point_clouds_pose_estimation_simple_vis_node = xarm6_3d_detection_and_tracking.inteld435i_point_clouds_pose_estimation_simple_vis_node:main',
            'azure_kinect_listener_node = xarm6_3d_detection_and_tracking.azure_kinect_listener_node:main',
            'azure_kinect_filtered_listener_node = xarm6_3d_detection_and_tracking.azure_kinect_filtered_listener_node:main',
            'azure_kinect_rgbd_image_generator_node = xarm6_3d_detection_and_tracking.azure_kinect_rgbd_image_generator_node:main',
            'azure_kinect_source_scanned_point_cloud = xarm6_3d_detection_and_tracking.azure_kinect_source_scanned_point_cloud:main',
            'azure_kinect_aligned_point_clouds = xarm6_3d_detection_and_tracking.azure_kinect_aligned_point_clouds:main',
            'perform_pose_reg_on_source_scanned = xarm6_3d_detection_and_tracking.perform_pose_reg_on_source_scanned:main',
            'pose_registration = xarm6_3d_detection_and_tracking.pose_registration:main',
        ],
    },
)