import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import struct

class RGBDImageGeneratedAzureKinectListenerNode(Node):
    def __init__(self):
        super().__init__('azure_kinect_source_point_cloud')
        self.bridge = CvBridge()
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/rgb/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 10)

        # PointCloud2 publisher
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/filtered_point_cloud', 10)
        
        # Image storage
        self.color_image = None
        self.depth_image = None
        
        self.min_component_area = 1000
        
    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_display()
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_display()
    
    def create_color_mask(self, color_image):
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lo_black = np.array([0, 0, 0])
        up_black = np.array([180, 255, 70])
        color_mask = cv2.inRange(hsv_img, lo_black, up_black)
        self.get_logger().info(
        f'Color Mask: {color_mask.shape}, {color_mask.dtype}')
        return color_mask
    
    def create_depth_mask(self, depth_image, target_shape):
        """Creating the Depth Mask"""
        lo_d = 350
        up_d = 525
        depth_mask = cv2.inRange(depth_image, lo_d, up_d)
        self.get_logger().info(
        f'Depth Mask: {depth_mask.shape}, {depth_mask.dtype}')
        return depth_mask
    
    def combine_masks(self, depth_mask, color_mask):
        return cv2.bitwise_and(depth_mask, color_mask)
    
    def clean_mask_with_connected_components(self, combined_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        clean_mask = np.zeros_like(combined_mask)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            if stats[largest_idx, cv2.CC_STAT_AREA] >= self.min_component_area:
                clean_mask[labels == largest_idx] = 255

        return clean_mask
    
    def generate_rgbd_image(self, combined_mask, depth_image, color_image):
        result_rgb = cv2.bitwise_and(color_image, color_image, mask=combined_mask)

        # Mask depth image safely with numpy to preserve uint16 values
        masked_depth = np.zeros_like(depth_image)
        masked_depth[combined_mask == 255] = depth_image[combined_mask == 255]
        result_depth = masked_depth

        o3d_color = o3d.geometry.Image(result_rgb.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(result_depth.astype(np.uint16))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

        return result_rgb, result_depth, rgbd_image
    
    def convert_o3d_to_ros2_pointcloud2(self, o3d_cloud, frame_id="map", timestamp=None):
        header = Header()
        header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
        header.frame_id = frame_id

        points = np.asarray(o3d_cloud.points)
        has_colors = o3d_cloud.has_colors()
        colors = np.asarray(o3d_cloud.colors) if has_colors else None

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]

        if has_colors:
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))

        point_step = 16 if has_colors else 12
        data = bytearray()

        for i in range(len(points)):
            x, y, z = points[i]
            data.extend(struct.pack('fff', x, y, z))

            if has_colors:
                r, g, b = (colors[i] * 255).astype(np.uint8)
                rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
                data.extend(struct.pack('f', rgb))

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = point_step
        pointcloud_msg.row_step = point_step * len(points)
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = data

        return pointcloud_msg
    
    def process_and_display(self):
        if self.color_image is None or self.depth_image is None:
            return

        try:
            # Resize depth image to color image size
            depth_resized = cv2.resize(self.depth_image, (self.color_image.shape[1], self.color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            color_mask = self.create_color_mask(self.color_image)
            depth_mask = self.create_depth_mask(depth_resized, self.color_image.shape[:2])

            combined_mask = self.combine_masks(depth_mask, color_mask)
            clean_combined_mask = self.clean_mask_with_connected_components(combined_mask)

            filtered_color, filtered_depth, rgbd_image = self.generate_rgbd_image(
                clean_combined_mask, depth_resized, self.color_image)

            # Normalize depth for display
            depth_display = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Show all masks and results
            self.get_logger().info(f"Color mask shape: {color_mask.shape}, Depth mask shape: {depth_mask.shape}")
            cv2.imshow('Original Color Image', self.color_image)
            cv2.imshow('Color Mask', color_mask)
            cv2.imshow('Depth Mask', depth_mask)
            cv2.imshow('Combined Mask (Raw)', combined_mask)
            cv2.imshow('Combined Mask (Cleaned)', clean_combined_mask)
            cv2.imshow('Filtered Color Image', filtered_color)
            cv2.imshow('Filtered Depth Image (Colormap)', depth_colormap)

            # Open3D Intrinsics (Azure Kinect)
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(
                width=self.color_image.shape[1], height=self.color_image.shape[0],
                fx=607.7908935546875, fy=607.75390625, cx=640.822509765625, cy=369.03350830078125)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
            
            # Translate point cloud to have its centroid at the origin
            centroid = pcd.get_center()
            pcd.translate(-centroid)

            # Display point cloud
            #o3d.visualization.draw_geometries([pcd])

            # 🖥️ Visualize intermediate steps for debugging
            # depth_display = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX)
            # depth_display = np.uint8(depth_display)
            # depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # cv2.imshow('Original Color Image', self.color_image)
            # cv2.imshow('Color Mask', color_mask)
            # cv2.imshow('Depth Mask', depth_mask)
            # cv2.imshow('Combined Mask (Raw)', combined_mask)
            # cv2.imshow('Combined Mask (Cleaned)', clean_combined_mask)
            # cv2.imshow('Filtered Color Image', filtered_color)
            # cv2.imshow('Filtered Depth Image (Colormap)', depth_colormap)

            # Publish to /filtered_point_cloud
            point_cloud_msg = self.convert_o3d_to_ros2_pointcloud2(pcd)

            if point_cloud_msg:
                self.point_cloud_pub.publish(point_cloud_msg)
                self.get_logger().info("Published point cloud to /filtered_point_cloud")
            else:
                self.get_logger().warn("Empty point cloud — nothing published.")

            self.get_logger().info("Processing cycle complete.")
            
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in processing: {str(e)}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RGBDImageGeneratedAzureKinectListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()