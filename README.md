# 📌 XARM6 3D Semi-Deformable Object Detection and Tracking Project

This project implements a point cloud-based pose estimation system using an **xArm6 robot in Gazebo**, an **Azure Kinect simulated depth and color sensor**, and **Open3D** for real-time point cloud processing and registration.

It enables the xArm6 to scan an **oil pan model** in simulation and estimate its pose by aligning a captured point cloud to a reference model. The system will later support grasping and manipulation.

---

## 📸 Scenario Overview

- Simulate an **xArm6 manipulator** with a **Realsense D435i** (or Azure Kinect equivalent) in Gazebo.
- Acquire **color and depth images**
- Process and mask the oil pan object
- Generate a **filtered point cloud**
- Align the scanned cloud to a reference model using **RANSAC + ICP**
- Visualize source and aligned target models in Open3D

---

## 📦 Dependencies

- ROS 2 Humble
- `xarm_ros2` workspace
- Gazebo
- `open3d`
- `cv_bridge`, `sensor_msgs`, `std_srvs`
- `numpy`, `opencv-python`

---

## 🚀 How to Use

### 1️⃣ Clone this project

```bash
git clone https://github.com/YOUR_USERNAME/MSRxarm6_3d_detection_and_tracking.git

### 2️⃣ Add oil pan model to Gazebo

To have the oil pan appear in the simulation, download the oilpan directory and copy it to your ~/.gazebo/models directory. Make sure to hit Ctrl + h in order to make the directory unhidden and navigate to it.

### 3️⃣ Build the workspace

```bash
cd ~/xarm6_3d_detection_and_tracking_ws/
colcon build
source install/setup.bash

### 4️⃣ Launch the Gazebo simulation

```bash
ros2 launch xarm_moveit_config xarm6_moveit_gazebo.launch.py add_realsense_d435i:=true

This will:
- Spawn the xArm6 robot.
- Add a simulated Realsense D435i sensor.
- Load the oil pan model in Gazebo.

### 5️⃣ Run the point cloud pose estimation node

```bash
source ~/xarm6_3d_detection_and_tracking_ws/install/setup.bash
ros2 run xarm6_3d_detection_and_tracking point_clouds_pose_estimation_node

### 6️⃣ (Optional) Capture a reference model point cloud

When your oil pan is positioned as a model reference:

```bash
ros2 service call /save_model_point_cloud std_srvs/srv/Empty

This will save the current point cloud as model_object.ply for alignment.