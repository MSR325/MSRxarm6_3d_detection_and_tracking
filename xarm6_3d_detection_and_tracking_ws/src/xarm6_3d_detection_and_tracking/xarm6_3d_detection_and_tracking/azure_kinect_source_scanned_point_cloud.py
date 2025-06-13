import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import struct

class RGBDPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('azure_kinect_source_scanned_point_cloud')
        self.bridge = CvBridge()
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/rgb/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 10)

        # PointCloud2 publisher
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/source_scanned_point_cloud', 10)
        
        # Image storage
        self.color_image = None
        self.depth_image = None
        
        self.min_component_area = 1000
        
        # Azure Kinect intrinsics
        self.fx = 607.7908935546875
        self.fy = 607.75390625
        self.cx = 640.822509765625
        self.cy = 369.03350830078125
        
        # Frame ID for point cloud
        self.frame_id = "camera_base"  # Change this to match your camera frame
        
        self.get_logger().info("RGBD Point Cloud Publisher initialized")
        
    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_publish()
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_publish()
    
    def create_color_mask(self, color_image):
        """Create mask based on color (black objects)"""
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lo_black = np.array([0, 0, 0])
        up_black = np.array([180, 255, 50])
        color_mask = cv2.inRange(hsv_img, lo_black, up_black)
        return color_mask
    
    def create_depth_mask(self, depth_image):
        """Create mask based on depth range"""
        lo_d = 350
        up_d = 525
        depth_mask = cv2.inRange(depth_image, lo_d, up_d)
        return depth_mask
    
    def combine_masks(self, depth_mask, color_mask):
        """Combine depth and color masks"""
        return cv2.bitwise_and(depth_mask, color_mask)
    
    def clean_mask_with_connected_components(self, combined_mask):
        """Clean mask by keeping only the largest connected component"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        clean_mask = np.zeros_like(combined_mask)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            if stats[largest_idx, cv2.CC_STAT_AREA] >= self.min_component_area:
                clean_mask[labels == largest_idx] = 255

        return clean_mask
    
    def generate_rgbd_image(self, combined_mask, depth_image, color_image):
        """Generate filtered RGBD image"""
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
    
    def create_pointcloud2_from_o3d(self, o3d_pcd, header):
        """Convert Open3D point cloud to ROS2 PointCloud2 message"""
        points = np.asarray(o3d_pcd.points)
        colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else None
        
        if len(points) == 0:
            self.get_logger().warn("Empty point cloud generated")
            return None
        
        if colors is not None and len(colors) > 0:
            # With colors - convert RGB to BGR for proper display
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            # Create point data with RGB
            point_data = []
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors_uint8[i]
                # Pack RGB into uint32, then convert to float32
                # Note: OpenCV uses BGR, but RViz expects RGB
                rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
                rgb_float = struct.unpack('f', struct.pack('I', rgb_packed))[0]
                point_data.append([x, y, z, rgb_float])
            
            points_array = np.array(point_data, dtype=np.float32)
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
        else:
            # No colors - just XYZ
            points_array = points.astype(np.float32)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
        
        # Create PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
        cloud_msg.is_dense = False  # May contain invalid points
        
        return cloud_msg
    
    def process_and_publish(self):
        """Main processing function"""
        if self.color_image is None or self.depth_image is None:
            return

        try:
            # Resize depth image to color image size
            height, width = self.color_image.shape[:2]
            depth_resized = cv2.resize(self.depth_image, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create masks
            color_mask = self.create_color_mask(self.color_image)
            depth_mask = self.create_depth_mask(depth_resized)

            # Combine and clean masks
            combined_mask = self.combine_masks(depth_mask, color_mask)
            clean_combined_mask = self.clean_mask_with_connected_components(combined_mask)

            # Generate RGBD image
            filtered_color, filtered_depth, rgbd_image = self.generate_rgbd_image(
                clean_combined_mask, depth_resized, self.color_image)

            # Create Open3D intrinsics
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(
                width=width, height=height,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

            # Generate point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            
            # Transform point cloud (Azure Kinect standard transform)
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
            
            # Center the point cloud at origin
            if len(pcd.points) > 0:
                centroid = pcd.get_center()
                pcd.translate(-centroid)
                
                # Log point cloud info
                points_np = np.asarray(pcd.points)
                min_bounds = np.min(points_np, axis=0)
                max_bounds = np.max(points_np, axis=0)
                
                self.get_logger().info(
                    f"Generated point cloud: {len(pcd.points)} points, "
                    f"bounds: X[{min_bounds[0]:.3f}, {max_bounds[0]:.3f}] "
                    f"Y[{min_bounds[1]:.3f}, {max_bounds[1]:.3f}] "
                    f"Z[{min_bounds[2]:.3f}, {max_bounds[2]:.3f}]",
                    throttle_duration_sec=2.0
                )
                
                # Create ROS2 message header
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.frame_id
                
                # Convert to PointCloud2 and publish
                cloud_msg = self.create_pointcloud2_from_o3d(pcd, header)
                if cloud_msg is not None:
                    self.point_cloud_pub.publish(cloud_msg)
                    self.get_logger().debug(f"Published point cloud to {self.frame_id}")
            
            # Optional: Display intermediate results (remove in production)
            self.display_debug_images(
                self.color_image, color_mask, depth_mask, 
                combined_mask, clean_combined_mask, 
                filtered_color, filtered_depth
            )

        except Exception as e:
            self.get_logger().error(f'Error in processing: {str(e)}')
    
    def display_debug_images(self, color_image, color_mask, depth_mask, 
                           combined_mask, clean_combined_mask, 
                           filtered_color, filtered_depth):
        """Display debug images (optional - comment out for production)"""
        try:
            # Normalize depth for display
            depth_display = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Show all masks and results
            #cv2.imshow('Original Color Image', color_image)
            #cv2.imshow('Color Mask', color_mask)
            #cv2.imshow('Depth Mask', depth_mask)
            #cv2.imshow('Combined Mask (Raw)', combined_mask)
            #cv2.imshow('Combined Mask (Cleaned)', clean_combined_mask)
            #cv2.imshow('Filtered Color Image', filtered_color)
            #cv2.imshow('Filtered Depth Image (Colormap)', depth_colormap)
            
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().debug(f"Debug display error: {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RGBDPointCloudPublisher()

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