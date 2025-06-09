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
            'azure_kinect_listener_node = xarm6_3d_detection_and_tracking.azure_kinect_listener_node:main',
            'filtered_azure_kinect_listener_node = xarm6_3d_detection_and_tracking.filtered_azure_kinect_listener_node:main',
            'rgbd_image_generated_azure_kinect_node = xarm6_3d_detection_and_tracking.rgbd_image_generated_azure_kinect_node:main',
            'point_clouds_pose_estimation_node = xarm6_3d_detection_and_tracking.point_clouds_pose_estimation_node:main',
            'point_clouds_pose_estimation_simple_vis_node = xarm6_3d_detection_and_tracking.point_clouds_pose_estimation_simple_vis_node:main',
        ],
    },
)
