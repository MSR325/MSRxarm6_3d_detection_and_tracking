import open3d as o3d
import numpy as np
import os

def debug_point_cloud_file(file_path):
    """Debug function to check point cloud file reading"""
    print(f"\n=== Debugging Point Cloud: {os.path.basename(file_path)} ===")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return None
    
    print(f"File exists: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    try:
        # Try to read the point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Check if points were loaded
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)
        
        print(f"Points loaded: {len(points)}")
        print(f"Colors loaded: {len(colors)}")
        print(f"Normals loaded: {len(normals)}")
        
        if len(points) == 0:
            print("WARNING: No points loaded from file!")
            
            # Try alternative reading methods
            print("\nTrying alternative reading methods...")
            
            # Try reading as mesh first
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)
                if len(np.asarray(mesh.vertices)) > 0:
                    print(f"File contains mesh with {len(np.asarray(mesh.vertices))} vertices")
                    # Convert mesh to point cloud
                    pcd_from_mesh = mesh.sample_points_poisson_disk(10000)
                    print(f"Converted to point cloud with {len(np.asarray(pcd_from_mesh.points))} points")
                    return pcd_from_mesh
            except Exception as e:
                print(f"Failed to read as mesh: {e}")
                
        else:
            print(f"Point cloud bounds:")
            print(f"  Min: {points.min(axis=0)}")
            print(f"  Max: {points.max(axis=0)}")
            print(f"  Center: {points.mean(axis=0)}")
            
            # Check for NaN or infinite values
            nan_count = np.isnan(points).sum()
            inf_count = np.isinf(points).sum()
            print(f"NaN values: {nan_count}")
            print(f"Infinite values: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                print("WARNING: Point cloud contains invalid values!")
                # Clean the point cloud
                valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
                if valid_mask.sum() > 0:
                    pcd.points = o3d.utility.Vector3dVector(points[valid_mask])
                    if len(colors) > 0:
                        pcd.colors = o3d.utility.Vector3dVector(colors[valid_mask])
                    print(f"Cleaned point cloud has {len(np.asarray(pcd.points))} valid points")
            
        return pcd
        
    except Exception as e:
        print(f"ERROR reading point cloud: {e}")
        return None

def test_registration_compatibility(source_pcd, target_pcd, voxel_size=1.0):
    """Test if point clouds are suitable for registration"""
    print(f"\n=== Testing Registration Compatibility ===")
    
    if source_pcd is None or target_pcd is None:
        print("ERROR: One or both point clouds are None")
        return False
    
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    print(f"Source points: {len(source_points)}")
    print(f"Target points: {len(target_points)}")
    
    # Minimum points check
    min_points_required = 100
    if len(source_points) < min_points_required:
        print(f"ERROR: Source has insufficient points ({len(source_points)} < {min_points_required})")
        return False
    
    if len(target_points) < min_points_required:
        print(f"ERROR: Target has insufficient points ({len(target_points)} < {min_points_required})")
        return False
    
    # Check point cloud bounds overlap
    source_bounds = [source_points.min(axis=0), source_points.max(axis=0)]
    target_bounds = [target_points.min(axis=0), target_points.max(axis=0)]
    
    print(f"Source bounds: {source_bounds}")
    print(f"Target bounds: {target_bounds}")
    
    # Check for reasonable scale
    source_scale = np.linalg.norm(source_bounds[1] - source_bounds[0])
    target_scale = np.linalg.norm(target_bounds[1] - target_bounds[0])
    
    print(f"Source scale: {source_scale:.3f}")
    print(f"Target scale: {target_scale:.3f}")
    
    if source_scale < 1e-6 or target_scale < 1e-6:
        print("WARNING: Very small point cloud scale detected")
    
    # Test downsampling
    try:
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        print(f"After downsampling (voxel_size={voxel_size}):")
        print(f"  Source: {len(np.asarray(source_down.points))} points")
        print(f"  Target: {len(np.asarray(target_down.points))} points")
        
        if len(np.asarray(source_down.points)) < 10 or len(np.asarray(target_down.points)) < 10:
            print("WARNING: Too few points after downsampling. Try smaller voxel_size.")
            return False
            
    except Exception as e:
        print(f"ERROR during downsampling: {e}")
        return False
    
    print("Point clouds appear suitable for registration")
    return True

# Test your point cloud files
oil_pan_front_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_front_pc_10000.ply"
oil_pan_full_pc_path = "/home/chrisrvt/Projects/MSRxarm6_3d_detection_and_tracking/xarm6_3d_detection_and_tracking_ws/src/xarm6_3d_detection_and_tracking/pointClouds/oil_pan_full_pc_10000.ply"

print("Testing point cloud file reading...")

# Debug both files
source_pcd = debug_point_cloud_file(oil_pan_front_pc_path)
target_pcd = debug_point_cloud_file(oil_pan_full_pc_path)

# Test registration compatibility
test_registration_compatibility(source_pcd, target_pcd, voxel_size=1.0)

# If point clouds loaded successfully, try a quick registration test
if source_pcd is not None and target_pcd is not None:
    print(f"\n=== Quick Registration Test ===")
    try:
        # Apply your initial transformation
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                                 [1.0, 0.0, 0.0, 0.0], 
                                 [0.0, 1.0, 0.0, 0.0], 
                                 [0.0, 0.0, 0.0, 1.0]])
        
        source_transformed = source_pcd.transform(trans_init)
        
        # Try basic ICP registration (fixed API call)
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 
            max_correspondence_distance=2.0,  # Fixed parameter name
            init=np.identity(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        print(f"Registration fitness: {result.fitness:.4f}")
        print(f"Registration RMSE: {result.inlier_rmse:.4f}")
        
        if result.fitness > 0.1:
            print("Registration appears to work!")
        else:
            print("Registration may have issues - low fitness score")
            
    except Exception as e:
        print(f"Registration test failed: {e}")

print(f"\n=== Recommendations ===")
print("1. Check that your ROS2 node is receiving point clouds with sufficient points")
print("2. Add debug prints in your ROS2 node to log point cloud sizes")
print("3. Consider adjusting voxel_size if downsampling removes too many points")
print("4. Verify that the point cloud filtering in your pipeline isn't too aggressive")