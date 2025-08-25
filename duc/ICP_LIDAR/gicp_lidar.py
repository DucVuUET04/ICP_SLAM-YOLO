import open3d as o3d
import numpy as np

def lidar_to_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud
def downsample_point_cloud(point_cloud, voxel_size):
    if not point_cloud.has_points():
        return point_cloud
    return point_cloud.voxel_down_sample(voxel_size)
def gicp(points1, points2, threshold=200,  voxel_size=20, trans_init=np.eye(4)):
    if len(points1) < 10 or len(points2) < 10:

        return float('inf'), np.eye(4)
    
    source_pcd = lidar_to_point_cloud(points1)
    target_pcd = lidar_to_point_cloud(points2)
    
    source_pcd = downsample_point_cloud(source_pcd, voxel_size)
    target_pcd = downsample_point_cloud(target_pcd, voxel_size)

    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))

    source_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
    target_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria)
    
    return reg_p2p.inlier_rmse, reg_p2p.transformation
def transform_points(points, rotation_matrix, translation_vector):
    if len(points) == 0:
        return points
    points = np.asarray(points)
    return np.dot(points, rotation_matrix.T) + translation_vector
