import open3d as o3d
import numpy as np
import copy
import time
import os 
import sys
import cv2
import json

# Sources
oil_pan_front_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_front_pc_10000.ply"
oil_pan_sv_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_sv_pc_10000.ply"
# Target
oil_pan_full_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"

# Giving them their names:
source_path = oil_pan_front_pc_path
target_path = oil_pan_full_pc_path

def draw_registration_result(source, target, transformation):
    """Visualize registration result"""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    window_name="Registration Result")

def preprocess_point_cloud(pcd, voxel_size):
    """Preprocess point cloud for registration"""
    print(":: Downsample with a voxel size of %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f":: After downsampling: {len(pcd_down.points)} points")

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    """Load and prepare point clouds for registration"""
    print(":: Load two point clouds and disturb the initial pose.")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    print(f":: Loaded source: {len(source.points)} points")
    print(f":: Loaded target: {len(target.points)} points")
    
    # Initial transformation to align coordinate frames
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                             [1.0, 0.0, 0.0, 0.0], 
                             [0.0, 1.0, 0.0, 0.0], 
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    
    # Save transformed source for debugging
    o3d.io.write_point_cloud("transformed_source_front.ply", source)
    print(":: Saved transformed source as 'transformed_source_front.ply'")
    
    # Show initial alignment
    print(":: Showing initial alignment...")
    draw_registration_result(source, target, np.identity(4))

    # Preprocess both point clouds
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Perform global registration using RANSAC"""
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f, " % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, 
        distance_threshold, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        3,  # ransac_n
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

def refine_registration(source, target, voxel_size, ransac_result):
    """Refine registration using ICP"""
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, 
        max_correspondence_distance=distance_threshold,  # Fixed parameter name
        init=ransac_result.transformation, 
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

def compute_normals(pcd, radius):
    """Compute normals for point cloud"""
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

def evaluate_registration(source, target, transformation, threshold=1.0):
    """Evaluate registration quality"""
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    
    # Compute distances between corresponding points
    distances = source_temp.compute_point_cloud_distance(target)
    distances = np.asarray(distances)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean(distances**2))
    fitness = np.sum(distances < threshold) / len(distances)
    
    print(f":: Registration evaluation:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Fitness: {fitness:.4f}")
    print(f"   Mean distance: {np.mean(distances):.4f}")
    print(f"   Max distance: {np.max(distances):.4f}")
    
    return rmse, fitness

# Main registration pipeline
if __name__ == "__main__":
    print("=== Oil Pan Registration Pipeline ===")
    
    # Parameters
    voxel_size = 1.0  # Adjust based on your point cloud scale
    
    try:
        # Step 1: Prepare dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
        
        # Step 2: Global registration
        print("\n=== Global Registration (RANSAC) ===")
        result_ransac = execute_global_registration(source_down, target_down, 
                                                  source_fpfh, target_fpfh, 
                                                  voxel_size)
        print("Global registration result:")
        print(result_ransac)
        print("Transformation matrix:")
        print(result_ransac.transformation)
        
        # Visualize global registration result
        print(":: Showing global registration result...")
        draw_registration_result(source_down, target_down, result_ransac.transformation)
        
        # Step 3: Refine with ICP
        print("\n=== Local Refinement (ICP) ===")
        
        # Compute normals for ICP
        radius_normal = voxel_size * 2
        compute_normals(source, radius_normal)
        compute_normals(target, radius_normal)
        
        result_icp = refine_registration(source, target, voxel_size, result_ransac)
        print("ICP registration result:")
        print(result_icp)
        print("Final transformation matrix:")
        print(result_icp.transformation)
        
        # Visualize final result
        print(":: Showing final registration result...")
        draw_registration_result(source, target, result_icp.transformation)
        
        # Step 4: Evaluate registration quality
        print("\n=== Registration Evaluation ===")
        rmse, fitness = evaluate_registration(source, target, result_icp.transformation)
        
        # Save results
        print("\n=== Saving Results ===")
        
        # Save final aligned source
        source_aligned = copy.deepcopy(source)
        source_aligned.transform(result_icp.transformation)
        o3d.io.write_point_cloud("final_aligned_source.ply", source_aligned)
        print(":: Saved aligned source as 'final_aligned_source.ply'")
        
        # Save transformation matrix
        np.savetxt("transformation_matrix.txt", result_icp.transformation, fmt='%.6f')
        print(":: Saved transformation matrix as 'transformation_matrix.txt'")
        
        # Save registration summary
        summary = {
            "voxel_size": voxel_size,
            "global_registration": {
                "fitness": float(result_ransac.fitness),
                "inlier_rmse": float(result_ransac.inlier_rmse),
                "correspondence_set_size": len(result_ransac.correspondence_set)
            },
            "local_registration": {
                "fitness": float(result_icp.fitness),
                "inlier_rmse": float(result_icp.inlier_rmse),
                "correspondence_set_size": len(result_icp.correspondence_set)
            },
            "final_evaluation": {
                "rmse": float(rmse),
                "fitness": float(fitness)
            },
            "transformation_matrix": result_icp.transformation.tolist()
        }
        
        with open("registration_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(":: Saved registration summary as 'registration_summary.json'")
        
        print(f"\n=== Registration Complete ===")
        print(f"Final fitness: {fitness:.4f}")
        print(f"Final RMSE: {rmse:.4f}")
        
        if fitness > 0.3:
            print("✅ Registration appears successful!")
        else:
            print("⚠️ Registration may need improvement - consider adjusting parameters")
            
    except Exception as e:
        print(f"❌ Registration failed with error: {e}")
        import traceback
        traceback.print_exc()