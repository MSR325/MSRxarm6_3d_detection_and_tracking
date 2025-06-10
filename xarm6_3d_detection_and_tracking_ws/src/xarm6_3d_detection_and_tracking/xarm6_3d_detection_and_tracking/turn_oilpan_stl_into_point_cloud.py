import open3d as o3d

oil_pan_full_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/models/OilPanBasic.stl"
oil_pan_front_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/models/OilPanFrontView.stl"
oil_pan_sv_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/models/OilPanSideView1.stl"

oil_pan_full = o3d.io.read_triangle_mesh(oil_pan_full_path)
oil_pan_front = o3d.io.read_triangle_mesh(oil_pan_front_path)
oil_pan_sv = o3d.io.read_triangle_mesh(oil_pan_sv_path)

number_of_points = 10000 # You can adjust this number

oil_pan_full_pc = oil_pan_full.sample_points_poisson_disk(number_of_points)
oil_pan_front_pc = oil_pan_front.sample_points_poisson_disk(number_of_points)
oil_pan_sv_pc = oil_pan_sv.sample_points_poisson_disk(number_of_points)

#o3d.visualization.draw_geometries([oil_pan_full_pc])

#o3d.visualization.draw_geometries([oil_pan_front_pc])

#o3d.visualization.draw_geometries([oil_pan_sv_pc])

oil_pan_full_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"
oil_pan_front_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_front_pc_10000.ply"
oil_pan_sv_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_sv_pc_10000.ply"

o3d.io.write_point_cloud(oil_pan_full_pc_path,oil_pan_full_pc)
o3d.io.write_point_cloud(oil_pan_front_pc_path,oil_pan_front_pc)
o3d.io.write_point_cloud(oil_pan_sv_pc_path,oil_pan_sv_pc)