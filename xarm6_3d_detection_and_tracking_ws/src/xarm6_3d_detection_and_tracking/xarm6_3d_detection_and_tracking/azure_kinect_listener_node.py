import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class AzureKinectListenerNode(Node):
    def __init__(self):
        super().__init__('azure_kinect_listener_node')

        # Bridge to convert ROS <-> OpenCV images
        self.bridge = CvBridge()

        # Subscribers
        self.color_sub = self.create_subscription(
            Image,
            '/color/image_raw',
            self.color_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # Initialize image holders
        self.color_image = None
        self.depth_image = None

    def color_callback(self, msg):
        # Convert to OpenCV image
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.show_images()

    def depth_callback(self, msg):
        # Convert to OpenCV image (16UC1 format)
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.show_images()

    def show_images(self):
        # Only display if both images are available
        if self.color_image is not None and self.depth_image is not None:
            # Normalize depth image for visualization (uint8)
            depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)

            # Apply a colormap
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Show images
            cv2.imshow('Color Image', self.color_image)
            cv2.imshow('Depth Image', depth_colormap)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = AzureKinectListenerNode()
    rclpy.spin(node)

    # Clean shutdown
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
