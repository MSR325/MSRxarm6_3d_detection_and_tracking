import open3d as o3d
import numpy as np

oil_pan_full_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"
oil_pan_front_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_front_pc_10000.ply"
oil_pan_sv_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_sv_pc_10000.ply"

def redefine_the_centroid(pc_path,re_centered_path):
    # Load original point cloud
    pcd = o3d.io.read_point_cloud(pc_path)
    # Compute centroid
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    # Create point cloud for the centroid
    centroid_pcd = o3d.geometry.PointCloud()
    centroid_pcd.points = o3d.utility.Vector3dVector([centroid])
    # Assign red color to the centroid
    #centroid_color = [1, 0, 0]  # RGB for red
    #centroid_pcd.colors = o3d.utility.Vector3dVector([centroid_color])
    # Create a coordinate frame (axes) at the origin (can be adjusted to another position)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # Visualize everything together
    o3d.visualization.draw_geometries([pcd, centroid_pcd, coord_frame])
    o3d.io.write_point_cloud(re_centered_path, pcd)

full_model_translated_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/centeredPointClouds/oil_pan_full_pc_10000.ply"
front_view_translated_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/centeredPointClouds/oil_pan_front_pc_10000.ply"
side_view_translated_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/centeredPointClouds/oil_pan_sv_pc_10000.ply"

original_pc_paths = [oil_pan_full_pc_path, oil_pan_front_pc_path, oil_pan_sv_pc_path]
translated_pc_paths = [full_model_translated_path, front_view_translated_path, side_view_translated_path]

for i in range(len(original_pc_paths)):
    redefine_the_centroid(original_pc_paths[i], translated_pc_paths[i])
print("Done!")