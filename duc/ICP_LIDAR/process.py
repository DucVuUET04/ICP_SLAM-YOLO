import open3d as o3d
import numpy as np
import cv2
import math
import os
import logging
import gicp_lidar
import Config
def load_and_prepare_scan(filepath):
    if not os.path.exists(filepath):
        print(f"Không tìm thấy tệp '{filepath}'")       
        logging.error(f"Không tìm thấy tệp '{filepath}'")
        return None
    
    try:
        scan_data = np.load(filepath)
        if scan_data.ndim != 2 or scan_data.shape[1] not in [2, 3]:                                        
            print(f"Định dạng không hợp lệ trong '{filepath}'. Shape: {scan_data.shape}")
            logging.error(f"Định dạng không hợp lệ trong '{filepath}'. Shape: {scan_data.shape}")
            return None
        
        scan_data = np.asarray(scan_data, dtype=np.float64)
        
        if scan_data.shape[1] == 3:
            points_3d = polar_to_cartesian_3d(scan_data)
        else:
            print(f"Phát hiện định dạng cartesian 2D trong '{filepath}'. Đang thêm trục Z...")
            logging.info(f"Phát hiện định dạng cartesian 2D trong '{filepath}'. Đang thêm trục Z...")
            z_column = np.zeros((scan_data.shape[0], 1))
            points_3d = np.hstack((scan_data, z_column))
        
        return points_3d
    except Exception as e:
        print(f"Lỗi khi tải tệp '{filepath}': {str(e)}")
        logging.error(f"Lỗi khi tải tệp '{filepath}': {str(e)}")
        return None
    
def polar_to_cartesian_3d(scan_data):
    if scan_data is None or len(scan_data) == 0:
        return np.array([])
    points_cartesian = []

    for point in scan_data:
        quality, angle, distance = point
        is_in_front_arc = (angle <= 135) or (angle >= 225)
        if distance > 1000 and distance < 9000 and quality > 10 and is_in_front_arc:
            angle_rad = math.radians(angle)
            x = distance * math.cos(angle_rad)
            y = -distance * math.sin(angle_rad)
            points_cartesian.append([x, y, 0.0])
 
    return np.array(points_cartesian)


def lidar_to_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud



def filter_outliers(point_cloud, nb_neighbors=35, std_ratio=1.0):
    if not point_cloud.has_points():
        return point_cloud
    pcd, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def remove_duplicate_points(points, voxel_size=20.0):
    if len(points) == 0:
        return np.array([])
    pcd = lidar_to_point_cloud(points)
    pcd_down = gicp_lidar.downsample_point_cloud(pcd, voxel_size)
    return np.asarray(pcd_down.points)

def remove_dynamic_points(current_points, prev_points, distance_threshold=250.0):
    if prev_points is None or len(prev_points) == 0 or len(current_points) == 0:
        return current_points
    pcd_current = lidar_to_point_cloud(current_points)
    pcd_prev = lidar_to_point_cloud(prev_points)
    distances = pcd_current.compute_point_cloud_distance(pcd_prev)
    distances = np.asarray(distances)
    static_indices = np.where(distances < distance_threshold)[0]
    static_pcd = pcd_current.select_by_index(static_indices)
    return np.asarray(static_pcd.points)

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def update_occupancy_map(occupancy_map, points_global, robot_pos, map_center_px, resolution,
                         p_occ_inc=0.2, p_free_dec=0.9,area = 140):
    if len(points_global) == 0:
        return
    
    h, w = occupancy_map.shape[:2]

    # if occupancy_map.ndim == 3 and occupancy_map.shape[2] == 3:
    if not hasattr(update_occupancy_map, "occupancy_probs"):
        update_occupancy_map.occupancy_probs = np.full((h, w), 0.5, dtype=np.float32)

    occ = update_occupancy_map.occupancy_probs
    # else:
        # occ = occupancy_map

    robot_x_px = int(map_center_px[0] + robot_pos[0] / resolution)
    robot_y_px = int(map_center_px[1] - robot_pos[1] / resolution)
    x1= max(0,robot_x_px - area )
    y1= max(0, robot_y_px - area)
    x2= min(w, robot_x_px +area)
    y2= min(h, robot_y_px +area )
    occupancy_map_new=occupancy_map[y1:y2, x1:x2,:]
    occ_new =update_occupancy_map.occupancy_probs[y1:y2, x1:x2]

    height, width = occupancy_map_new.shape[:2]
        
    robot_x_px_new =robot_x_px -x1
    robot_y_px_new =robot_y_px -y1


    for pt in points_global:
        px = int(map_center_px[0] + pt[0] / resolution-x1)
        py = int(map_center_px[1] - pt[1] / resolution -y1)
        
        if not (0 <= px < width and 0 <= py < height):
            continue
        
        line = bresenham_line(robot_x_px_new, robot_y_px_new, px, py)
      
        for i, (x, y) in enumerate(line):
            if not (0 <= x < width and 0 <= y < height):
                continue
            
            threshold_down = 0.15
            threshold_up= 0.65
           
            # if occ_new[y,x]>= threshold_up:
            #     break
            if i == len(line) - 1:
                occ_new[y, x] = min(1.0, occ_new[y, x] + p_occ_inc)
            else :
                if occ_new[y,x]>= threshold_up:
                    break
                occ_new[y, x] = max(0.0, occ_new[y, x] * p_free_dec)
            # if occ_new[y, x] < threshold_down:
            #     occ_new[y, x] = 0
    
   
    occ_uint8 = ((1 - occ_new) * 255).astype(np.uint8)
    occupancy_map_new[:, :, 0] = occ_uint8
    occupancy_map_new[:, :, 1] = occ_uint8
    occupancy_map_new[:, :, 2] = occ_uint8
    occupancy_map[y1:y2, x1:x2,:]=  occupancy_map_new
    update_occupancy_map.occupancy_probs[y1:y2, x1:x2] = occ_new
    # print(occupancy_map_new.shape)
    # cv2.imshow("mini map",occupancy_map_new)
def scan_on_map(occupancy_map, points, map_center_px, resolution, color=(0, 255, 0)):
    for point in points:
        point_x_px = int(map_center_px[0] + point[0] / resolution)
        point_y_px = int(map_center_px[1] - point[1] / resolution)
        if 0 <= point_x_px < occupancy_map.shape[1] and 0 <= point_y_px < occupancy_map.shape[0]:
            cv2.circle(occupancy_map, (point_x_px, point_y_px), 1, color, -1)

def draw_robot_pose(occupancy_map, pose, map_center_px, resolution, axis_length=300):
    position = pose[:3, 3]
    rotation = pose[:3, :3]

    robot_x_px = int(map_center_px[0] + position[0] / resolution)
    robot_y_px = int(map_center_px[1] - position[1] / resolution)
    robot_center_px = (robot_x_px, robot_y_px)

    x_axis_vec = np.array([axis_length, 0, 0])
    x_axis_end_vec = rotation @ x_axis_vec
    x_end_px = (int(robot_x_px + x_axis_end_vec[0] / resolution), int(robot_y_px - x_axis_end_vec[1] / resolution))
    
    cv2.arrowedLine(occupancy_map, robot_center_px, x_end_px, (0, 0, 255), 1, tipLength=0.3)
    cv2.circle(occupancy_map, robot_center_px, 5, (255, 0, 0), -1)
  

def filter_new_points_by_occupancy(points_to_add, occupancy_probs, map_center_px, resolution, free_threshold=0.2):
    """Lọc các điểm mới dựa trên bản đồ chiếm dụng hiện có.
    Loại bỏ các điểm rơi vào các ô được coi là không gian trống để giảm nhiễu.
    """
    if len(points_to_add) == 0 or occupancy_probs is None:
        return points_to_add

    height, width = occupancy_probs.shape
    
    indices_to_keep = []
    for i, point in enumerate(points_to_add):
        px = int(map_center_px[0] + point[0] / resolution)
        py = int(map_center_px[1] - point[1] / resolution)

        if not (0 <= px < width and 0 <= py < height):
            indices_to_keep.append(i)
            continue

        if occupancy_probs[py, px] < free_threshold:
            continue 
        
        indices_to_keep.append(i)

    return points_to_add[indices_to_keep]

def prune_global_map(global_map, occupancy_probs, map_center_px, resolution,
                     free_threshold=0.2):
    #lọc những điểm nằm ngoài vùng và những điểm có tỉ lệ chiếm dụng nhỏ hơn ngưỡng 
    if len(global_map.points) == 0 or occupancy_probs is None:
        return global_map

    points = np.asarray(global_map.points)
    height, width = occupancy_probs.shape
    
    indices_to_keep = []
    for i, point in enumerate(points):
        px = int(map_center_px[0] + point[0] / resolution)
        py = int(map_center_px[1] - point[1] / resolution)

        if not (0 <= px < width and 0 <= py < height):
            indices_to_keep.append(i)
            continue
        if occupancy_probs[py, px] < free_threshold:
            continue 
        indices_to_keep.append(i)

    return global_map.select_by_index(indices_to_keep)

def draw_target_point(occupancy_map, target_point, map_center_px, resolution, color=(255, 255, 0)):
    """Draws a target point on the occupancy map."""
    target_x_px = int(map_center_px[0] + target_point[0] / resolution)
    target_y_px = int(map_center_px[1] - target_point[1] / resolution)
    if 0 <= target_x_px < occupancy_map.shape[1] and 0 <= target_y_px < occupancy_map.shape[0]:
        cv2.circle(occupancy_map, (target_x_px, target_y_px), 5, color, -1)

def calculate_distance(robot_pos, target_point):
    """Calculates the distance between the robot and the target point."""
    dx = target_point[0] - robot_pos[0]
    dy = target_point[1] - robot_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance
