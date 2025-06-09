# 📌 XARM6 3D Semi-Deformable Object Detection and Tracking Project

This project implements a point cloud-based pose estimation system using an **xArm6 robot in Gazebo**, an **Azure Kinect simulated depth and color sensor**, and **Open3D** for real-time point cloud processing and registration.

It enables the xArm6 to scan an **oil pan model** in simulation and estimate its pose by aligning a captured point cloud to a reference model. The system will later support grasping and manipulation.

---

## 📸 Scenario Overview

- Simulate an **xArm6 manipulator** with a **Realsense D435i** in Gazebo.
- Acquire **color and depth images**
- Process and mask the oil pan object
- Generate a **filtered point cloud**
- Align the scanned cloud to a reference model using **RANSAC + ICP**
- Visualize source and aligned target models in Open3D

- Visualize RGB, depth, and Point Cloud captured views from the **Azure Kinect** sensor.
- Acquire **color and depth images** and **Point Cloud**
- Process and mask the oil pan object
- Generate a **canonical model target point cloud**
- Align the **source scanned Point Cloud** to the **target canonical point cloud** of the oil pan using **RANSAC + ICP**
- Visualize aligned source and target models in Open3D.

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
```

### 2️⃣ Add oil pan model to Gazebo

To have the oil pan appear in the simulation, download the oilpan directory and copy it to your `~/.gazebo/models` directory. Make sure to hit `Ctrl + h` in order to make the directory unhidden and navigate to it.

### 3️⃣ Build the workspace

```bash
cd ~/xarm6_3d_detection_and_tracking_ws/
colcon build
source install/setup.bash
```

### 4️⃣ Launch the Gazebo simulation

```bash
ros2 launch xarm_moveit_config xarm6_moveit_gazebo.launch.py add_realsense_d435i:=true
```

This will:
- Spawn the xArm6 robot.
- Add a simulated Realsense D435i sensor.
- Load the oil pan model in Gazebo.

### 5️⃣ Run the point cloud pose estimation node

```bash
source ~/xarm6_3d_detection_and_tracking_ws/install/setup.bash
ros2 run xarm6_3d_detection_and_tracking point_clouds_pose_estimation_node
```

### 6️⃣ (Optional) Capture a reference model point cloud

When your oil pan is positioned as a model reference:

```bash
ros2 service call /save_model_point_cloud std_srvs/srv/Empty
```

This will save the current point cloud as `model_object.ply` for alignment.

## 📷 Setting up Azure Kinect on Ubuntu 22.04 with ROS 2 Humble

If you want to use a **physical Azure Kinect DK** sensor with this project or test your point cloud nodes against real sensor data, follow these steps:

### ✅ Install Azure Kinect SDK

The Microsoft Kinect for Azure (k4a) is only supported officially for Ubuntu 18.04.

In the following README.md, you can check the instructions to install its library in a more recent operating system:

https://gist.github.com/jlblancoc/ae2a082b0ed5af2e71645b04b7207210

The instructions are:

```bash
# Download these two .deb files: 
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb

# Install them:
sudo apt install ./*.deb
```

If this doesn't work, first download the required packages:

https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/
https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/
https://mirrors.edge.kernel.org/ubuntu/pool/universe/libs/libsoundio/

Then install them just like in the following procedure:

```bash
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt install libglvnd0 libusb-1.0-0
sudo dpkg -i libk4a1.4_1.4.1_amd64.deb libk4a1.4-dev_1.4.1_amd64.deb libk4abt1.1_1.1.2_amd64.deb libk4abt1.1-dev_1.1.2_amd64.deb libsoundio1_1.0.2-1_amd64.deb k4a-tools_1.4.1_amd64.deb
```

Create the rules.d file to allow opening the camera without "root" permissions:

- Download https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/scripts/99-k4a.rules
- Copy it into '/etc/udev/rules.d/'

To initialize the Azure Kinect Viewer, use the following command:

```bash
k4aviewer
```

### ✅ Download the Azure Kinect ROS2 Driver

Check the following links:

https://github.com/microsoft/Azure_Kinect_ROS_Driver/tree/humble
https://github.com/ckennedy2050/Azure_Kinect_ROS2_Driver

Clone and build the `humble` branch from https://github.com/microsoft/Azure_Kinect_ROS_Driver.git

```bash
cd ~/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/
git clone https://github.com/microsoft/Azure_Kinect_ROS_Driver.git -b humble
cd Azure_Kinect_ROS_Driver
colcon build --symlink-install --packages-select azure_kinect_ros_driver
source install/setup.bash
```

### ✅ Fixing building errors from the azure_kinect_ros_driver

If you encounter the following error while building:

```vbnet
error: no matching function for call to ‘rclcpp::Duration::Duration(double)’
```

**Fix it by editing the file:**
`Azure_Kinect_ROS_Driver/src/k4a_ros_device.cpp`
Go to line **771**:
Replace:

```cpp
marker_msg->lifetime = rclcpp::Duration(0.25);
```

With:

```cpp
marker_msg->lifetime = rclcpp::Duration::from_seconds(0.25);
```

Save the file and rebuild:

```bash
cd ~/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/Azure_Kinect_ROS_Driver
colcon build --symlink-install --packages-select azure_kinect_ros_driver
source install/setup.bash
```

### ✅ Visualize Azure Kinect Topics in Rviz2

Once your Azure Kinect driver is running:

```bash
./install/azure_kinect_ros_driver/lib/azure_kinect_ros_driver/node   --ros-args   -p fps:=30   -p color_enabled:=true   -p rgb_point_cloud:=true
```

Then, start RViz2:

```bash
rviz2
```

When inside, you can select 4 fixed frames provided by the **Azure Kinect** sensor:
1. camera_base
2. depth_camera_link
3. imu_link
4. rgb_camera_link

To create a visualization, you can select by topic the following options:

1. /depth
    /image_raw
        Camera
            raw
            compressed
            compressedDepth
        DepthCloud
        Image
2. /depth_to_rgb
    /image_raw
        Camera
            raw
            compressed
            compressedDepth
        DepthCloud
        Image
3. /ir
    /image_raw
    Camera
        raw
        compressed
        compressedDepth
    DepthCloud
    Image
4. /points2
    PointCloud2
5. /rgb
    /image_raw
        Camera
            raw
            compressed
            compressedDepth
        DepthCloud
        Image
6. /rgb_to_depth
    /image_raw
        Camera
            raw
            compressed
            compressedDepth
        DepthCloud
        Image

The `Camera` option displays an image from the camera, with the visualized world rendered behind it. The `DepthCloud` option displays point clouds based on depth maps. The `Image` display creates a new rendering window in an image. 


- Add a PointCloud2 display
- Set the topic to `/depth/points`
- You should now see your live point cloud stream from the Kinect sensor.

## 📎 Notes

- The system is designed for ROS 2 Humble on Ubuntu 22.04.
- Point clouds can be aligned using Open3D’s RANSAC and ICP implementation.
- Both simulation and real-sensor support via Azure Kinect or Realsense D435i are possible.