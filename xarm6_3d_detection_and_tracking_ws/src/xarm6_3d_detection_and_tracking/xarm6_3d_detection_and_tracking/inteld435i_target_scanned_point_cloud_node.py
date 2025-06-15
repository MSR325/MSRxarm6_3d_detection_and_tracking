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
        super().__init__('inteld435i_target_scanned_point_cloud_node')
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

        # PointCloud2 publisher
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/scanned_model_point_cloud', 10)
        
        # Image storage
        self.color_image = None
        self.depth_image = None
        
        # Parameters for filtering
        self.min_component_area = 1000  # Minimum area to keep connected components
        
    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_display()
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_display()
    
    def create_color_mask(self, color_image):
        """Algorithm 5: Creating the Color Mask"""
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lo_black = np.array([0, 0, 0])
        up_black = np.array([180, 255, 70])
        color_mask = cv2.inRange(hsv_img, lo_black, up_black)
        return color_mask
    
    def create_depth_mask(self, depth_image):
        """Creating the Depth Mask"""
        lo_d = 350
        up_d = 600 # 525
        depth_mask = cv2.inRange(depth_image, lo_d, up_d)
        return depth_mask
    
    def combine_masks(self, depth_mask, color_mask):
        """Algorithm 6: Combining the Masks"""
        # Both masks are combined using bitwise AND operation
        combined_mask = cv2.bitwise_and(depth_mask, color_mask)
        return combined_mask
    
    def clean_mask_with_connected_components(self, combined_mask):
        """Clean the combined mask using connected components analysis"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        
        # Create clean mask
        clean_mask = np.zeros_like(combined_mask)
        
        if num_labels > 1:  # Background is label 0, so we need at least 2 labels
            # Find the largest component (excluding background at index 0)
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background component
            
            if len(areas) > 0:
                # Get the index of the largest component (add 1 to account for skipping background)
                largest_component_idx = np.argmax(areas) + 1
                
                # Only keep the largest component if it's above minimum area threshold
                if stats[largest_component_idx, cv2.CC_STAT_AREA] >= self.min_component_area:
                    clean_mask[labels == largest_component_idx] = 255
        
        return clean_mask
    
    def generate_rgbd_image(self, combined_mask, depth_image, color_image):
        """Algorithm 7: Generating the RGB-D Image"""
        # Color image is passed through the combined mask
        result_rgb = cv2.bitwise_and(color_image, color_image, mask=combined_mask)
        
        # Depth image is masked as well
        result_depth = cv2.bitwise_and(depth_image, depth_image, mask=combined_mask)
        
        # Create RGB-D image using Open3D
        # Convert to Open3D format
        o3d_color = o3d.geometry.Image(result_rgb.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(result_depth.astype(np.uint16))
        
        # Create RGB-D image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth,
            depth_scale=1000.0,  # Adjust based on your depth scale
            depth_trunc=3.0,     # Truncate depth beyond 3 meters
            convert_rgb_to_intensity=False
        )
        
        return result_rgb, result_depth, rgbd_image
    
    def process_and_display(self):
        if self.color_image is None or self.depth_image is None:
            return

        try:
            # Step 1: Create individual masks
            color_mask = self.create_color_mask(self.color_image)
            depth_mask = self.create_depth_mask(self.depth_image)

            # Step 2: Combine masks using bitwise AND
            combined_mask = self.combine_masks(depth_mask, color_mask)

            # Step 3: Clean the combined mask using connected components
            clean_combined_mask = self.clean_mask_with_connected_components(combined_mask)

            # Step 4: Generate RGB-D image
            filtered_color, filtered_depth, rgbd_image = self.generate_rgbd_image(
                clean_combined_mask, self.depth_image, self.color_image
            )

            # # Prepare depth visualization
            # depth_display = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX)
            # depth_display = np.uint8(depth_display)
            # depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # # Display results
            # cv2.imshow('Original Color Image', self.color_image)
            # cv2.imshow('Color Mask', color_mask)
            # cv2.imshow('Depth Mask', depth_mask)
            # cv2.imshow('Combined Mask (Raw)', combined_mask)
            # cv2.imshow('Combined Mask (Cleaned)', clean_combined_mask)
            # cv2.imshow('Filtered Color Image', filtered_color)
            # cv2.imshow('Filtered Depth Image', depth_colormap)

            # # Optional: Display mask statistics
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            #     combined_mask, connectivity=8
            # )

            # if num_labels > 1:
            #     areas = stats[1:, cv2.CC_STAT_AREA]
            #     self.get_logger().info(
            #         f'Connected Components: {num_labels-1}, '
            #         f'Areas: {areas.tolist()}, '
            #         f'Largest: {np.max(areas) if len(areas) > 0 else 0}'
            #     )

            # Open3D Intrinsic Parameters (Azure Kinect)
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(
                width=self.color_image.shape[1],
                height=self.color_image.shape[0],
                fx=607.7908935546875,
                fy=607.75390625,
                cx=640.822509765625,
                cy=369.03350830078125
            )

            # Generate Point Cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsics
            )

            # Flip the point cloud to align with OpenCV convention if necessary
            pcd.transform([[1, 0,  0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0,  0, 1]])

            # Visualize Point Cloud
            #o3d.visualization.draw_geometries([pcd])

            # Visualize Point Cloud on RViz2
            pc2_msg = self.convert_open3d_to_pointcloud2(pcd)
            self.point_cloud_pub.publish(pc2_msg)

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in processing: {str(e)}')

    def convert_open3d_to_pointcloud2(self, pcd):
        # Convert Open3D PointCloud to PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_link"  # Adjust to your RViz frame

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        data = []
        for i in range(len(points)):
            x, y, z = points[i]
            if len(colors) > 0:
                r, g, b = (colors[i] * 255).astype(np.uint8)
            else:
                r, g, b = 255, 255, 255  # default white
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            data.append(struct.pack('ffff', x, y, z, struct.unpack('f', struct.pack('I', rgb))[0]))

        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.fields = fields
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16  # 4*4 bytes
        pc2_msg.row_step = pc2_msg.point_step * len(points)
        pc2_msg.is_dense = True
        pc2_msg.data = b''.join(data)

        return pc2_msg

    
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