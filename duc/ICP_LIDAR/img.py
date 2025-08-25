import numpy as np

def stereo_to_3d(points_left, points_right, f, cx, cy, B):
    points_left = np.array(points_left, dtype=float)
    points_right = np.array(points_right, dtype=float)

    disparity = np.abs(points_left[:,0] - points_right[:,0])
    disparity[disparity == 0] = 1e-6

    Z = (f * B) / disparity
    X = ((points_left[:,0] - cx) * Z) / f
    Y = ((points_left[:,1] - cy) * Z) / f

    return np.vstack((X, Y, Z)).T

def pallet_orientation_and_distance(corners_3d):
    """
    Tính toán hướng và khoảng cách của pallet.

    Args:
        corners_3d: Mảng numpy chứa tọa độ 3D của các góc pallet.
        holes_3d: Mảng numpy chứa tọa độ 3D của các lỗ trên pallet.
    Returns:
        normal, angle_deg, mean_depth, dist_hole: Pháp tuyến, góc nghiêng, khoảng cách trung bình, khoảng cách tới lỗ.
    """
    v1 = corners_3d[1] - corners_3d[0]
    v2 = corners_3d[2] - corners_3d[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    if normal[2] < 0:
        normal = -normal

    angle_yaw_rad = np.arctan2(normal[0], normal[2])


    mean_depth = np.mean(corners_3d[:,2])

    return normal, angle_yaw_rad, mean_depth