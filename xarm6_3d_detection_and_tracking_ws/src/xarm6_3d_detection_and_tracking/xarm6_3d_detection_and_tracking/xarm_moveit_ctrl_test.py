#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from copy import deepcopy
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetCartesianPath
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
import tf_transformations

class XArmMoveItController(Node):
    def __init__(self):
        super().__init__('xarm_moveit_controller')
        self.action_client = ActionClient(self, MoveGroup, 'move_action')
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/xarm6_traj_controller/follow_joint_trajectory')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6"
        ]
        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.get_logger().info("xArm MoveIt controller initialized.")

    def joint_state_callback(self, msg):
        joint_pos_dict = dict(zip(msg.name, msg.position))
        self.current_joint_state = [joint_pos_dict.get(j, 0.0) for j in self.joint_names]

    def get_end_effector_pose(self):
        if self.current_joint_state is None:
            self.get_logger().warn("No joint state received yet.")
            return None

        fk_client = self.create_client(GetPositionFK, '/compute_fk')
        if not fk_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("FK service not available")
            return None

        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = 'base_link'
        fk_request.fk_link_names = ['link6']
        fk_request.robot_state.joint_state.name = self.joint_names
        fk_request.robot_state.joint_state.position = self.current_joint_state

        future = fk_client.call_async(fk_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            poses = future.result().pose_stamped
            if poses:
                return poses[0].pose
        else:
            self.get_logger().error("FK service call failed")
            return None

    def plan_and_execute(self, goal_rad):
        if self.current_joint_state is None:
            self.get_logger().warn("Waiting for current joint state...")
            return None, None

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available.")
            return None, None

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = "xarm6"
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        constraints = Constraints()
        for name, position in zip(self.joint_names, goal_rad):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraints)

        self.get_logger().info("Sending joint goal to MoveIt...")
        goal_future = self.action_client.send_goal_async(goal_msg)
        return goal_future, None

    def plan_and_execute_pose(self, pose_goal):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available.")
            return None, None

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = "xarm6"
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        constraints = Constraints()
        from moveit_msgs.msg import PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive

        pc = PositionConstraint()
        pc.link_name = 'link6'
        pc.header.frame_id = pose_goal.header.frame_id
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = [0.01, 0.01, 0.01]
        pc.constraint_region.primitives.append(shape)
        pc.constraint_region.primitive_poses.append(pose_goal.pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.link_name = 'link6'
        oc.header.frame_id = pose_goal.header.frame_id
        oc.orientation = pose_goal.pose.orientation
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0

        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        goal_msg.request.goal_constraints.append(constraints)

        self.get_logger().info("Sending pose goal to MoveIt...")
        goal_future = self.action_client.send_goal_async(goal_msg)
        return goal_future, None


    def execute_cartesian_path(self, start_pose: PoseStamped):
        waypoints = []

        radius = 0.1  # 5 cm radius
        center_x = start_pose.pose.position.x
        center_y = start_pose.pose.position.y
        center_z = start_pose.pose.position.z
        orientation = deepcopy(start_pose.pose.orientation)

        num_points = 40  # more = smoother circle

        for i in range(num_points):
            theta = 2 * math.pi * (i / num_points)
            pose = PoseStamped()
            pose.header.frame_id = "link_base"
            pose.pose.position.x = center_x + radius * math.cos(theta)
            pose.pose.position.y = center_y + radius * math.sin(theta)
            pose.pose.position.z = center_z  # constant height
            pose.pose.orientation = orientation  # keep end-effector orientation fixed
            waypoints.append(pose.pose)
            
        final_pose = PoseStamped()
        final_pose.header.frame_id = "link_base"
        final_pose.pose.position.x = center_x
        final_pose.pose.position.y = center_y
        final_pose.pose.position.z = center_z
        final_pose.pose.orientation = orientation
        waypoints.append(final_pose.pose)    


        req = GetCartesianPath.Request()
        req.group_name = "xarm6"
        req.header.frame_id = "link_base"
        req.start_state.joint_state.name = self.joint_names
        req.start_state.joint_state.position = self.current_joint_state
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if not result or not result.solution.joint_trajectory.points:
            self.get_logger().error("Cartesian path planning failed.")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = result.solution.joint_trajectory

        self.get_logger().info("Sending Cartesian trajectory...")
        send_future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory was rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result.result.error_code == 0:
            self.get_logger().info("✅ Cartesian path executed successfully.")
        else:
            self.get_logger().error(f"❌ Trajectory execution failed with code {result.result.error_code}")


def main(args=None):
    rclpy.init(args=args)
    node = XArmMoveItController()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.current_joint_state is None:
                print(" Waiting for current joint state...")
                continue

            print(f" Current Joint Angles (deg): {[f'{math.degrees(r):.2f}' for r in node.current_joint_state]}")
            pose = node.get_end_effector_pose()
            if pose:
                print(f" EE Position: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
                q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                r, p, y = tf_transformations.euler_from_quaternion(q)
                print(f" EE Orientation (RPY, deg): roll={math.degrees(r):.1f}°, pitch={math.degrees(p):.1f}°, yaw={math.degrees(y):.1f}°")

            mode = ""
            while mode not in ["1", "2", "3"]:
                print("\nSelect control mode:")
                print("1. Send Joint Angles")
                print("2. Send Pose Goal")
                print("3. Execute Circular Trajectory")
                mode = input("Enter 1, 2 or 3: ").strip()

            if mode == "1":
                input_str = input("Enter GOAL joint angles (6 values in degrees): ").strip().strip('"').strip("'")
                try:
                    goal_deg = [float(val) for val in input_str.split()]
                    if len(goal_deg) != 6:
                        print(" Please enter exactly 6 values.")
                        continue
                    goal_rad = [math.radians(d) for d in goal_deg]
                    goal_future, _ = node.plan_and_execute(goal_rad)
                except ValueError:
                    print(" Invalid input. Enter 6 space-separated numbers.")
                    continue

            elif mode == "2":
                pose = PoseStamped()
                pose.header.frame_id = 'base_link'
                try:
                    xyz = input("Enter x y z in meters (e.g., 0.3 0 0.4): ").strip().split()
                    rpy = input("Enter roll pitch yaw in degrees (e.g., 0 180 0): ").strip().split()
                    if len(xyz) != 3 or len(rpy) != 3:
                        print(" Please enter 3 values for position and 3 for orientation.")
                        continue

                    pose.pose.position.x = float(xyz[0])
                    pose.pose.position.y = float(xyz[1])
                    pose.pose.position.z = float(xyz[2])

                    r, p, y = [math.radians(float(a)) for a in rpy]
                    q = tf_transformations.quaternion_from_euler(r, p, y)
                    pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

                    goal_future, _ = node.plan_and_execute_pose(pose)

                except ValueError:
                    print(" Invalid input. Make sure all values are numbers.")
                    continue

            elif mode == "3":
                pose = node.get_end_effector_pose()
                if pose:
                    print(" Executing circular trajectory from current pose...")
                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = "base_link"
                    pose_stamped.pose = pose
                    node.execute_cartesian_path(pose_stamped)
                else:
                    print(" No valid current pose.")
                    continue

            while rclpy.ok() and 'goal_future' in locals() and not goal_future.done():
                rclpy.spin_once(node, timeout_sec=0.1)

            if 'goal_future' in locals():
                goal_handle = goal_future.result()
                if not goal_handle.accepted:
                    node.get_logger().error(" Goal was rejected by MoveIt.")
                    continue

                result_future = goal_handle.get_result_async()
                while rclpy.ok() and not result_future.done():
                    rclpy.spin_once(node, timeout_sec=0.1)

                result = result_future.result().result
                if result.error_code.val == result.error_code.SUCCESS:
                    node.get_logger().info(" Trajectory executed successfully!")
                else:
                    node.get_logger().error(f" Planning failed with error code: {result.error_code.val}")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
