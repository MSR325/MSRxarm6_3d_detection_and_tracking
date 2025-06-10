# Usage Instructions:

## 1. Launch File

The `azure_kinect_oil_pan_detection_and_tracking.launch.py` launch file will start all three nodes in the correct order and establish the data flow from Azure Kinect → Point Cloud Generation → Registration. The RViz configuration will show the raw scan, canonical model, and registered result simultaneously for easy verification of the alignment quality.

## 2. Update Package Names

Replace `'xarm6_3d_detection_and_tracking'` with your actual ROS2 package name in the launch file.

## 3. Create RViz Config Directory

```bash
mkdir -p ~/your_workspace/src/your_package/config/
```

Save the RViz configuration as `oil_pan_registration.rviz` in the config/ directory.

## 4. Launch the Complete System

Basic launch:

```bash
ros2 launch xarm6_3d_detection_and_tracking azure_kinect_oil_pan_detection_and_tracking.launch.py
```

With custom parameters:

```bash
ros2 launch your_package_name azure_kinect_oil_pan_detection_and_tracking.launch.py \
    canonical_model_path:="/path/to/your/oil_pan_full_pc_10000.ply" \
    voxel_size:=0.5 \
    registration_frequency:=3.0 \
    color_resolution:=1080P \
    fps:=30
```

With RViz visualization:

```bash
ros2 launch xarm6_3d_detection_and_tracking azure_kinect_oil_pan_detection_and_tracking.launch.py use_rviz:=true
```

## 5. Monitor the System

Check if all nodes are running:

```bash
ros2 node list
```

Monitor topics:

```bash
ros2 topic list
ros2 topic echo /registered_point_cloud
```

Check TF transforms:

```bash
ros2 run tf2_tools view_frames
```