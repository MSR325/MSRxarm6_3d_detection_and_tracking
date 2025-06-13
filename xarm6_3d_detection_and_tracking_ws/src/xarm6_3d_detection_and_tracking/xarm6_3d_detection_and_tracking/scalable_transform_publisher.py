import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import math

class ScalableTransformPublisher(Node):
    def __init__(self):
        super().__init__('scalable_transform_publisher')
        
        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Parameters
        self.declare_parameter('parent_frame', 'map')
        self.declare_parameter('child_frame', 'scaled_rotated_frame')
        self.declare_parameter('publish_rate', 10.0)
        
        # Transform parameters
        self.declare_parameter('translation_x', 0.0)
        self.declare_parameter('translation_y', 0.0)
        self.declare_parameter('translation_z', 0.0)
        
        self.declare_parameter('rotation_x', 0.0)  # Roll in degrees
        self.declare_parameter('rotation_y', 0.0)  # Pitch in degrees
        self.declare_parameter('rotation_z', 0.0)  # Yaw in degrees
        
        self.declare_parameter('scale_x', 1.0)
        self.declare_parameter('scale_y', 1.0)
        self.declare_parameter('scale_z', 1.0)
        
        # Animation parameters
        self.declare_parameter('animate_rotation', False)
        self.declare_parameter('rotation_speed', 1.0)  # degrees per second
        self.declare_parameter('rotation_axis', 'z')   # 'x', 'y', or 'z'
        
        # Get parameters
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        self.translation = [
            self.get_parameter('translation_x').value,
            self.get_parameter('translation_y').value,
            self.get_parameter('translation_z').value
        ]
        
        self.rotation_degrees = [
            self.get_parameter('rotation_x').value,
            self.get_parameter('rotation_y').value,
            self.get_parameter('rotation_z').value
        ]
        
        self.scale = [
            self.get_parameter('scale_x').value,
            self.get_parameter('scale_y').value,
            self.get_parameter('scale_z').value
        ]
        
        self.animate_rotation = self.get_parameter('animate_rotation').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.rotation_axis = self.get_parameter('rotation_axis').value
        
        # Animation state
        self.animation_angle = 0.0
        
        # Subscriber for dynamic transform updates
        self.transform_sub = self.create_subscription(
            Float64MultiArray,
            '/transform_params',
            self.transform_params_callback,
            10
        )
        
        # Timer for publishing transforms
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_transform)
        
        self.get_logger().info(f"Transform publisher started:")
        self.get_logger().info(f"  Parent frame: {self.parent_frame}")
        self.get_logger().info(f"  Child frame: {self.child_frame}")
        self.get_logger().info(f"  Translation: {self.translation}")
        self.get_logger().info(f"  Rotation (deg): {self.rotation_degrees}")
        self.get_logger().info(f"  Scale: {self.scale}")
        self.get_logger().info(f"  Animate rotation: {self.animate_rotation}")
        
        if self.animate_rotation:
            self.get_logger().info(f"  Animation axis: {self.rotation_axis}, speed: {self.rotation_speed} deg/s")
    
    def transform_params_callback(self, msg):
        """Update transform parameters dynamically
        Expected format: [tx, ty, tz, rx, ry, rz, sx, sy, sz]
        """
        if len(msg.data) >= 9:
            self.translation = [msg.data[0], msg.data[1], msg.data[2]]
            self.rotation_degrees = [msg.data[3], msg.data[4], msg.data[5]]
            self.scale = [msg.data[6], msg.data[7], msg.data[8]]
            
            self.get_logger().info(
                f"Updated transform - T: {self.translation}, "
                f"R: {self.rotation_degrees}, S: {self.scale}",
                throttle_duration_sec=1.0
            )
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # Convert degrees to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        
        # Calculate quaternion components
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        return [x, y, z, w]
    
    def apply_scaling_to_translation(self, translation, scale):
        """Apply scaling to translation (for non-uniform scaling effect)"""
        return [translation[i] * scale[i] for i in range(3)]
    
    def publish_transform(self):
        """Publish the transform with scaling and rotation"""
        try:
            # Create transform message
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = self.parent_frame
            transform.child_frame_id = self.child_frame
            
            # Apply scaling to translation (optional - creates scaling effect)
            scaled_translation = self.apply_scaling_to_translation(self.translation, self.scale)
            
            # Set translation
            transform.transform.translation.x = scaled_translation[0]
            transform.transform.translation.y = scaled_translation[1]
            transform.transform.translation.z = scaled_translation[2]
            
            # Calculate rotation
            rotation = self.rotation_degrees.copy()
            
            # Add animation if enabled
            if self.animate_rotation:
                # Update animation angle
                dt = 1.0 / self.publish_rate
                self.animation_angle += self.rotation_speed * dt
                if self.animation_angle >= 360.0:
                    self.animation_angle -= 360.0
                
                # Add animated rotation to the specified axis
                if self.rotation_axis == 'x':
                    rotation[0] += self.animation_angle
                elif self.rotation_axis == 'y':
                    rotation[1] += self.animation_angle
                else:  # 'z'
                    rotation[2] += self.animation_angle
            
            # Convert to quaternion
            quat = self.euler_to_quaternion(rotation[0], rotation[1], rotation[2])
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]
            
            # Note: ROS transforms don't directly support scaling
            # Scaling is typically handled by the nodes using the transform
            # or by publishing additional scale information
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {str(e)}")
    
    def get_current_transform_matrix(self):
        """Get the current 4x4 transformation matrix including scale"""
        # Translation matrix
        T = np.eye(4)
        T[0:3, 3] = self.translation
        
        # Rotation matrix
        roll, pitch, yaw = [math.radians(r) for r in self.rotation_degrees]
        
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(roll), -math.sin(roll), 0],
            [0, math.sin(roll), math.cos(roll), 0],
            [0, 0, 0, 1]
        ])
        
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch), 0],
            [0, 1, 0, 0],
            [-math.sin(pitch), 0, math.cos(pitch), 0],
            [0, 0, 0, 1]
        ])
        
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0, 0],
            [math.sin(yaw), math.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Combined rotation (ZYX order)
        R = Rz @ Ry @ Rx
        
        # Scale matrix
        S = np.eye(4)
        S[0, 0] = self.scale[0]
        S[1, 1] = self.scale[1]
        S[2, 2] = self.scale[2]
        
        # Combined transformation: T * R * S
        transform_matrix = T @ R @ S
        
        return transform_matrix

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ScalableTransformPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()