import open3d as o3d
import numpy as np
import cv2
import time
import math
import os



# --- PHẦN 1: CẤU HÌNH VÀ TẢI DỮ LIỆU ---

class Config:
    BASE_PATH = r"Scan_data_1\Scan_data_{}.npy"  
    OUTPUT_PCD = "global_map_offline.pcd"
    OUTPUT_OCCUPANCY_MAP = "realtime_occupancy_map.png"
    START_FILE = 1 
    END_FILE = 1500
    MAP_SIZE_MM = 25000
    RESOLUTION_MM_PER_PIXEL = 30
    MAP_WIDTH_MM = 30000  
    MAP_HEIGHT_MM = 25000 
    ICP_VOXEL_SIZE = 20.0
    ICP_THRESHOLD = 200.0
    MAX_RMSE_THRESHOLD = 50.0
    OUTLIER_NB_NEIGHBORS = 30
    OUTLIER_STD_RATIO =1.5
    DUPLICATE_VOXEL_SIZE = 30.0
    DYNAMIC_DISTANCE_THRESHOLD = 250.0
    ROBOT_AXIS_LENGTH_MM = 500
    DELAY_MS = 1
    IMG_SIZE = 800
    MAP_WIDTH_PIXELS = int(MAP_WIDTH_MM / RESOLUTION_MM_PER_PIXEL)
    MAP_HEIGHT_PIXELS = int(MAP_HEIGHT_MM / RESOLUTION_MM_PER_PIXEL)
    LOCAL_MAP_RADIUS_MM = 10000.0

def load_and_prepare_scan(filepath):
    if not os.path.exists(filepath):
        print(f"Không tìm thấy tệp '{filepath}'")
        return None
    
    try:
        scan_data = np.load(filepath)
        if scan_data.ndim != 2 or scan_data.shape[1] not in [2, 3]:
            print(f"Định dạng không hợp lệ trong '{filepath}'. Shape: {scan_data.shape}")
            return None
        
        scan_data = np.asarray(scan_data, dtype=np.float64)
        
        if scan_data.shape[1] == 3:
            points_3d = polar_to_cartesian_3d(scan_data)
        else:
            print(f"Phát hiện định dạng cartesian 2D trong '{filepath}'. Đang thêm trục Z...")
            z_column = np.zeros((scan_data.shape[0], 1))
            points_3d = np.hstack((scan_data, z_column))
        
        return points_3d
    except Exception as e:
        print(f"Lỗi khi tải tệp '{filepath}': {str(e)}")
        return None

# --- PHẦN 2: CÁC HÀM SLAM ---

def polar_to_cartesian_3d(scan_data):
    if scan_data is None or len(scan_data) == 0:
        return np.array([])
    points_cartesian = []
    for point in scan_data:
        quality, angle, distance = point
        is_in_front_arc = (angle <= 135) or (angle >= 225)
        if distance > 0 and distance < 10000 and quality > 13 and is_in_front_arc:
            angle_rad = math.radians(angle)
            x = distance * math.cos(angle_rad)
            y = -distance * math.sin(angle_rad)
            points_cartesian.append([x, y, 0.0])
    return np.array(points_cartesian)

def lidar_to_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def downsample_point_cloud(point_cloud, voxel_size):
    return point_cloud.voxel_down_sample(voxel_size)

def filter_outliers(point_cloud, nb_neighbors=35, std_ratio=1.0):
    pcd, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def remove_duplicate_points(points, voxel_size=40.0):
    pcd = lidar_to_point_cloud(points)
    pcd_down = downsample_point_cloud(pcd, voxel_size)
    return np.asarray(pcd_down.points)

def remove_dynamic_points(current_points, prev_points, distance_threshold=250.0):
    if prev_points is None or len(prev_points) == 0:
        return current_points 
    pcd_current = lidar_to_point_cloud(current_points)
    pcd_prev = lidar_to_point_cloud(prev_points)
    distances = pcd_current.compute_point_cloud_distance(pcd_prev)
    distances = np.asarray(distances)
    static_indices = np.where(distances < distance_threshold)[0]
    static_pcd = pcd_current.select_by_index(static_indices)
    return np.asarray(static_pcd.points)

def transform_points(points, rotation_matrix, translation_vector):
    points = np.asarray(points)
    return np.dot(points, rotation_matrix.T) + translation_vector

def gicp(points1, points2, threshold=200,  voxel_size=20, trans_init=np.eye(4)):
    ''' 
        voxel_size_old = 0.1 | đang test là 2: tạo 1 khung vuông 2x2 và gộp các điểm trong khung vuông
        max_iteration: số lần lặp tối đa
        max_nn + radius: số điểm tối đa max_nn nằm trong bán kính radius được xét (radius = 0.1(old) or 1)

        radius=1, max_nn=30, max_iteration=1000, voxel_size=2  --> tb 2.74
        radius=0.1, max_nn=30, max_iteration=1000, voxel_size=0.1 --> tb 2.79

        Tham số threshold chính là khoảng cách xa nhất mà một cặp điểm có thể được coi là một "cặp tương ứng hợp lệ" (inlier).
        có thể điều chỉnh threshold = 5
        
    '''
    source_pcd = lidar_to_point_cloud(points1)
    target_pcd = lidar_to_point_cloud(points2)

    source_pcd = downsample_point_cloud(source_pcd, voxel_size)
    target_pcd = downsample_point_cloud(target_pcd, voxel_size)


    # Tính toán các vector pháp tuyến cho PointClouds
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

    # Tính toán các ma trận hiệp phương sai cho PointCloud
    source_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    target_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

    # Thiết lập các tham số cho GICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)

    # Sử dụng GICP để căn chỉnh các điểm quét
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria)

    return reg_p2p.inlier_rmse, reg_p2p.transformation
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
            threshold_up= 0.75
           
           
            if i == len(line) - 1:
                occ_new[y, x] = min(1.0, occ_new[y, x] + p_occ_inc)
            else:
                # if occ_new[y,x]>= threshold_up:
                #     break
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
    cv2.imshow("nn",occupancy_map_new)
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
    #y_axis_vec = np.array([0, axis_length, 0])

    x_axis_end_vec = rotation @ x_axis_vec
    #y_axis_end_vec = rotation @ y_axis_vec

    x_end_px = (int(robot_x_px + x_axis_end_vec[0] / resolution), int(robot_y_px - x_axis_end_vec[1] / resolution))
    # y_end_px = (int(robot_x_px + y_axis_end_vec[0] / resolution), int(robot_y_px - y_axis_end_vec[1] / resolution))
    
    cv2.line(occupancy_map, robot_center_px, x_end_px, (0, 0, 255), 2)
    #cv2.line(occupancy_map, robot_center_px, y_end_px, (255, 0, 0), 2) 
    cv2.arrowedLine(occupancy_map, robot_center_px, x_end_px, (0, 0, 255), 1, tipLength=0.3)
    cv2.circle(occupancy_map, robot_center_px, 5, (255, 0, 0), -1)

# --- PHẦN 3: CHƯƠNG TRÌNH CHÍNH ---

if __name__ == "__main__":
    occupancy_map = np.full((Config.MAP_HEIGHT_PIXELS, Config.MAP_WIDTH_PIXELS, 3), 128, dtype=np.uint8)
    map_center_px = (Config.MAP_WIDTH_PIXELS // 2, Config.MAP_HEIGHT_PIXELS // 2)
   
    global_map = o3d.geometry.PointCloud()
    global_pose = np.eye(4)
    prev_points_global = None
    current_points_global = np.array([])
    pose_history = [] 
 

    print("\nBắt đầu xử lý dữ liệu từ file và lập bản đồ. Nhấn 'q' hoặc ESC để thoát.")

    try:
        
        first_scan_points = load_and_prepare_scan(Config.BASE_PATH.format(Config.START_FILE))
        if first_scan_points is None or len(first_scan_points) == 0:
            
            print("Lỗi: Không thể tải scan đầu tiên. Thoát chương trình.")
            exit()

        global_map.points.extend(o3d.utility.Vector3dVector(first_scan_points))
        current_points_global = first_scan_points
        update_occupancy_map(occupancy_map, current_points_global, global_pose[:3, 3], map_center_px, Config.RESOLUTION_MM_PER_PIXEL)
        pose_history.append(global_pose[:3, 3][:2].copy())

        for i in range(Config.START_FILE + 1, Config.END_FILE):
            scan_file = Config.BASE_PATH.format(i)
            scan_data = load_and_prepare_scan(scan_file)

            if scan_data is None or len(scan_data) == 0:
                print(f"Lỗi tải tệp {scan_file} hoặc tệp rỗng. Bỏ qua.")
                continue

            current_points = scan_data

            if len(current_points) < 10:
                continue

            # pcd_current = lidar_to_point_cloud(current_points)
            # pcd_current = filter_outliers(pcd_current, nb_neighbors=Config.OUTLIER_NB_NEIGHBORS, std_ratio=Config.OUTLIER_STD_RATIO)
            # current_points = np.asarray(pcd_current.points)
            if len(current_points) < 10:
              
                continue

            # Xử lý SLAM
            current_robot_pos = global_pose[:3, 3]
            all_map_points = np.asarray(global_map.points)
            
            if len(all_map_points) > 0:
                distances_sq = np.sum((all_map_points[:, :2] - current_robot_pos[:2])**2, axis=1)
                nearby_indices = np.where(distances_sq < Config.LOCAL_MAP_RADIUS_MM**2)[0]
                map_points_for_icp = all_map_points[nearby_indices]

                if len(map_points_for_icp) < 50:
                    map_points_for_icp = all_map_points
            else:
                map_points_for_icp = all_map_points
            
            map_for_display = occupancy_map.copy()
           
            
   
            rmse, transformation_matrix = gicp(current_points, map_points_for_icp, Config.ICP_THRESHOLD, Config.ICP_VOXEL_SIZE, trans_init=global_pose)

            print(f"RMSE: {rmse:.4f}")
            # scan_on_map(map_for_display,map_points_for_icp, map_center_px, Config.RESOLUTION_MM_PER_PIXEL)
            if rmse > Config.MAX_RMSE_THRESHOLD:
                continue
                print(f"Cảnh báo: RMSE cao ({rmse:.4f}) tại tệp {scan_file}, pose không được cập nhật.")
                # current_points_global = transform_points(current_points, global_pose[:3, :3], global_pose[:3, 3])
            else:
                global_pose = transformation_matrix
                current_points_global = transform_points(current_points, global_pose[:3, :3], global_pose[:3, 3])
              
                # points_to_add = remove_duplicate_points(current_points_global, voxel_size=Config.DUPLICATE_VOXEL_SIZE)
                points_to_add = remove_dynamic_points(current_points_global, prev_points_global, Config.DYNAMIC_DISTANCE_THRESHOLD)

                # --- BƯỚC LỌC MỚI THEO YÊU CẦU ---
                # Lọc các điểm dựa trên bản đồ chiếm dụng trước khi thêm vào bản đồ toàn cục.
                # Điều này giúp loại bỏ các điểm nhiễu rơi vào vùng đã được xác định là không gian trống.
                if hasattr(update_occupancy_map, "occupancy_probs"):
                    points_to_add = filter_new_points_by_occupancy(
                        points_to_add,
                        update_occupancy_map.occupancy_probs,
                        map_center_px,
                        Config.RESOLUTION_MM_PER_PIXEL
                    )

                if len(points_to_add) > 0:
                    global_map.points.extend(o3d.utility.Vector3dVector(points_to_add))
            if len(global_map.points) > 1000:
                    global_map = downsample_point_cloud(global_map, Config.ICP_VOXEL_SIZE  )


            current_pos = global_pose[:3, 3]
            pose_history.append(current_pos[:2].copy())  # Lưu x, y
    
            prev_points_global = current_points_global.copy()
            robot_pos_map = global_pose[:3, 3]
            update_occupancy_map(occupancy_map, current_points_global, robot_pos_map, map_center_px, Config.RESOLUTION_MM_PER_PIXEL)

            if hasattr(update_occupancy_map, "occupancy_probs"):
                num_points_before = len(global_map.points)
                global_map = prune_global_map(
                    global_map,
                    update_occupancy_map.occupancy_probs,
                    map_center_px,
                    Config.RESOLUTION_MM_PER_PIXEL
                )
                num_points_after = len(global_map.points)
           
            map_for_display = occupancy_map.copy()
            # scan_on_map(map_for_display, map_points_for_icp, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, color=(255, 0, 0))
            scan_on_map(map_for_display, current_points_global, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, color=(0, 255, 0))
            draw_robot_pose(map_for_display, global_pose, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, Config.ROBOT_AXIS_LENGTH_MM)
            cv2.imshow("Real-time SLAM Map", map_for_display)

      
            key = cv2.waitKey(Config.DELAY_MS)
            if key in [ord('q'), 27]: 
                break

    except KeyboardInterrupt:
        print("Người dùng dừng chương trình bằng Ctrl+C.")

    finally:
        if len(global_map.points) > 0:
            print("\nĐang xử lý và lưu bản đồ cuối cùng...")
            final_map = downsample_point_cloud(global_map, voxel_size=Config.ICP_VOXEL_SIZE)
            output_dir = os.path.dirname(Config.OUTPUT_PCD)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            o3d.io.write_point_cloud(Config.OUTPUT_PCD, final_map)
            print(f"✅ Đã lưu bản đồ toàn cục vào '{Config.OUTPUT_PCD}'")
            cv2.imwrite(Config.OUTPUT_OCCUPANCY_MAP, occupancy_map)
            print(f"✅ Đã lưu bản đồ chiếm dụng vào '{Config.OUTPUT_OCCUPANCY_MAP}'")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\nKhông có đủ dữ liệu để tạo bản đồ.")