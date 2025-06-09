import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class FilteredAzureKinectListenerNode(Node):
    def __init__(self):
        super().__init__('filtered_azure_kinect_listener_node')

        self.bridge = CvBridge()

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

        self.color_image = None
        self.depth_image = None

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_display()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_display()

    def process_and_display(self):
        if self.color_image is None or self.depth_image is None:
            return

        # --- Color Filtering (Black Pixels) ---
        hsv_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)
        lo_black = np.array([0, 0, 0])
        up_black = np.array([180, 255, 70])

        black_mask = cv2.inRange(hsv_img, lo_black, up_black)
        color_img_res = cv2.bitwise_and(self.color_image, self.color_image, mask=black_mask)

        # --- Depth Filtering (350mm-525mm) ---
        lo_d = 350
        up_d = 525

        d_mask = cv2.inRange(self.depth_image, lo_d, up_d)
        depth_res = cv2.bitwise_and(self.depth_image, self.depth_image, mask=d_mask)

        # Normalize depth for display
        depth_display = cv2.normalize(depth_res, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = np.uint8(depth_display)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Show results
        cv2.imshow('Original Color Image', self.color_image)
        cv2.imshow('Filtered Color Image (Black Range)', color_img_res)
        cv2.imshow('Filtered Depth Image (350mm-525mm)', depth_colormap)
        cv2.imshow('Black Mask', black_mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FilteredAzureKinectListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
