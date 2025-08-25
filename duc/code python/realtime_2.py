import open3d as o3d
import numpy as np
from rplidar import RPLidar, RPLidarException
import time
import serial.tools.list_ports
import threading
import cv2
import math
import logging
import serial

# --- PHẦN 1: LỚP ĐIỀU KHIỂN LIDAR ---

def get_com_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def check_com_port(port_name):
    return port_name in get_com_ports()

def auto_detect_lidar_port():
    for port in get_com_ports():
        try:
            lidar = RPLidar(port, 256000, timeout=0.1)
            info = lidar.get_info()
            lidar.disconnect()
            logging.info(f"Tìm thấy LiDAR trên cổng {port}: {info}")
            return port
        except (RPLidarException, serial.serialutil.SerialException) as e:
            logging.warning(f"Thử cổng {port} thất bại: {e}")
            continue
    return None

class LidarScanner:
    def __init__(self, port="COM6", baudrate=256000):
        self.port = port if port else auto_detect_lidar_port()
        self.baudrate = baudrate
        self.lidar = None
        self.latest_scan = None
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._initialized = False

    def connect(self, max_attempts=5, retry_delay=3):
        if not self.port:
            logging.error("Không tìm thấy cổng LiDAR khả dụng.")
            print("Lỗi: Không tìm thấy cổng LiDAR khả dụng.")
            return False
        for attempt in range(max_attempts):
            if not check_com_port(self.port):
                logging.error(f"Không tìm thấy cổng {self.port}. Các cổng có sẵn: {get_com_ports()}")
                print(f"Lỗi: Không tìm thấy cổng {self.port}. Các cổng có sẵn: {get_com_ports()}")
                return False
            try:
                logging.info(f"Thử kết nối tới LiDAR trên cổng {self.port} (lần {attempt + 1}/{max_attempts})...")
                print(f"Thử kết nối tới LiDAR trên cổng {self.port} (lần {attempt + 1}/{max_attempts})...")
                self.lidar = RPLidar(self.port, self.baudrate, timeout=5)
                self.lidar.reset()  
                time.sleep(1)
                info = self.lidar.get_info()
                self.lidar.start_motor()
                time.sleep(2) 
                health = self.lidar.get_health()
                if health[0] != 'Good':
                    logging.error(f"LiDAR không ở trạng thái tốt: {health}")
                    print(f"Lỗi: LiDAR không ở trạng thái tốt: {health}")
                    self.lidar.disconnect()
                    self.lidar = None
                    continue
                logging.info(f"Kết nối LiDAR thành công. Thông tin: {info}")
                print(f"✅ Kết nối LiDAR thành công. Thông tin: {info}")
                self._initialized = True
                return True
            except (RPLidarException, serial.serialutil.SerialException) as e:
                logging.error(f"Lỗi khi kết nối LiDAR: {e}")
                print(f"Lỗi khi kết nối LiDAR: {e}")
                if self.lidar:
                    self.lidar.disconnect()
                    self.lidar = None
                if attempt < max_attempts - 1:
                    time.sleep(retry_delay)
                    continue
                return False

    def start(self):
        if self.lidar is None and not self.connect():
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logging.info("Bắt đầu luồng đọc dữ liệu LiDAR.")
        print("🌀 Bắt đầu luồng đọc dữ liệu LiDAR.")

    def stop(self):
        logging.info("Đang dừng LiDAR...")
        print("🔴 Đang dừng LiDAR...")
        self._running = False
        if self.lidar is not None:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            except Exception as e:
                logging.error(f"Lỗi khi dừng LiDAR: {e}")
                print(f"Lỗi khi dừng LiDAR: {e}")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._initialized = False
        self.lidar = None
        logging.info("LiDAR đã dừng.")
        print("✅ LiDAR đã dừng.")

    def _read_loop(self):
        while self._running:
            try:
                if not self.check_health():
                    logging.error("LiDAR không ở trạng thái tốt, thử kết nối lại...")
                    print("LiDAR không ở trạng thái tốt, thử kết nối lại...")
                    self._initialized = False
                    self.connect()
                    continue
                for scan in self.lidar.iter_scans(min_len=10, max_buf_meas=1000):
                    if not self._running:
                        break
                    with self._lock:
                        self.latest_scan = np.array(scan)
                        logging.debug(f"Nhận được quét mới với {len(self.latest_scan)} điểm.")
            except (RPLidarException, serial.serialutil.SerialException) as e:
                logging.error(f"Lỗi trong luồng đọc LiDAR: {e}. Thử kết nối lại sau 3s...")
                print(f"Lỗi trong luồng đọc LiDAR: {e}. Thử kết nối lại sau 3s...")
                self._initialized = False
                time.sleep(3)
                if self._running:
                    self.connect()

    def get_scan(self):
        with self._lock:
            return self.latest_scan.copy() if self.latest_scan is not None else None

    def is_initialized(self):
        return self._initialized

    def check_health(self):
        try:
            return self.lidar.get_health()[0] == 'Good'
        except Exception as e:
            logging.error(f"Lỗi khi kiểm tra sức khỏe LiDAR: {e}")
            return False

# --- PHẦN 2: CÁC HÀM SLAM ---

def polar_to_cartesian_3d(scan_data):
    if scan_data is None or len(scan_data) == 0:
        return np.array([])
    points_cartesian = []
    for point in scan_data:
        quality, angle, distance = point
        #is_in_front_arc = (angle <= 135) or (angle >= 225)
        if distance > 0 and distance < 5000 and quality > 5:# and is_in_front_arc:
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

def filter_outliers(point_cloud, nb_neighbors=10, std_ratio=2.5):
    pcd, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def remove_duplicate_points(points, voxel_size=20.0):
    pcd = lidar_to_point_cloud(points)
    pcd_down = downsample_point_cloud(pcd, voxel_size)
    return np.asarray(pcd_down.points)

def remove_dynamic_points(current_points, prev_points, distance_threshold=150.0):
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

def gicp(points1, points2, threshold, voxel_size, trans_init=np.eye(4)):
    """
    Thực hiện thuật toán Generalized-ICP (GICP) để căn chỉnh hai đám mây điểm.
    Đây là phiên bản đã được sửa lỗi để sử dụng đúng TransformationEstimationForGeneralizedICP.
    """
    if len(points1) < 10 or len(points2) < 10:
        logging.warning("Không đủ điểm để chạy GICP.")
        return float('inf'), np.eye(4)

    source_pcd = lidar_to_point_cloud(points1)
    target_pcd = lidar_to_point_cloud(points2)

    # Tính toán các vector pháp tuyến cho PointCloud
    # Bán kính tìm kiếm nên liên quan đến kích thước voxel để đảm bảo đủ điểm lân cận
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Tính toán các ma trận hiệp phương sai cho PointCloud
    source_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Thiết lập các tham số cho GICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)

    # LỖI NGHIÊM TRỌNG Ở ĐÂY: Đang dùng ICP điểm-tới-điểm thay vì GICP
    # SỬA LỖI: Sử dụng TransformationEstimationForGeneralizedICP()
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
                         p_occ_inc=0.2, p_free_dec=0.9):
    if len(points_global) == 0:
        return

    height, width = occupancy_map.shape[:2]

    if occupancy_map.ndim == 3 and occupancy_map.shape[2] == 3:
        if not hasattr(update_occupancy_map, "occupancy_probs"):
            update_occupancy_map.occupancy_probs = np.full((height, width), 0.5, dtype=np.float32)  # bắt đầu ở 0.5

        occ = update_occupancy_map.occupancy_probs
    else:
        occ = occupancy_map

    robot_x_px = int(map_center_px[0] + robot_pos[0] / resolution)
    robot_y_px = int(map_center_px[1] - robot_pos[1] / resolution)

    for pt in points_global:
        px = int(map_center_px[0] + pt[0] / resolution)
        py = int(map_center_px[1] - pt[1] / resolution)
       
        if not (0 <= px < width and 0 <= py < height):
            continue

        line = bresenham_line(robot_x_px, robot_y_px, px, py)
      
        for i, (x, y) in enumerate(line):
            if not (0 <= x < width and 0 <= y < height):
                continue
        
            threshold_down = 0.15
            threshold_up= 0.75
            if occ[y,x]>= threshold_up:
                break
            if i == len(line) - 1:
                occ[y, x] = min(1.0, occ[y, x] + p_occ_inc)
            else:
                occ[y, x] = max(0.0, occ[y, x] * p_free_dec)
            if occ[y,x] < threshold_down:
                occ[y, x] = 0
            

    if occupancy_map.ndim == 3 and occupancy_map.shape[2] == 3:
        occ_uint8 = ((1-occ) * 255).astype(np.uint8)
        occupancy_map[:, :, 0] = occ_uint8 
        occupancy_map[:, :, 1] = occ_uint8 
        occupancy_map[:, :, 2] = occ_uint8 
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
    y_axis_vec = np.array([0, axis_length, 0])

    x_axis_end_vec = rotation @ x_axis_vec
    y_axis_end_vec = rotation @ y_axis_vec

    x_end_px = (int(robot_x_px + x_axis_end_vec[0] / resolution), int(robot_y_px - x_axis_end_vec[1] / resolution))
    y_end_px = (int(robot_x_px + y_axis_end_vec[0] / resolution), int(robot_y_px - y_axis_end_vec[1] / resolution))

    cv2.line(occupancy_map, robot_center_px, x_end_px, (0, 0, 255), 2)
    cv2.line(occupancy_map, robot_center_px, y_end_px, (255, 0, 0), 2) 

    cv2.circle(occupancy_map, robot_center_px, 5, (0, 255, 0), -1)
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


# --- PHẦN 3: CHƯƠNG TRÌNH CHÍNH KẾT HỢP ---

if __name__ == "__main__":
    LIDAR_PORT = "COM6"
    
    MAP_SIZE_MM = 5000
    RESOLUTION_MM_PER_PIXEL = 5
    MAP_DIM_PIXELS = int(MAP_SIZE_MM / RESOLUTION_MM_PER_PIXEL)
    ICP_VOXEL_SIZE = 20.0
    ICP_THRESHOLD = 130.0
    MIN_SCAN_INTERVAL = 0.5
    MAX_LIDAR_DISTANCE = 3000
    MIN_LIDAR_QUALITY = 5
    OUTLIER_NB_NEIGHBORS = 10
    OUTLIER_STD_RATIO = 2.5
    DUPLICATE_VOXEL_SIZE = 10.0
    DYNAMIC_DISTANCE_THRESHOLD = 150.0
    ROBOT_AXIS_LENGTH_MM = 200 

    # MAP_SIZE_MM = 20000
    # RESOLUTION_MM_PER_PIXEL = 20
    # MAP_DIM_PIXELS = int(MAP_SIZE_MM / RESOLUTION_MM_PER_PIXEL)
    # ICP_VOXEL_SIZE = 20.0
    # ICP_THRESHOLD = 200.0
    # MIN_SCAN_INTERVAL = 0.5
    # MAX_LIDAR_DISTANCE = 15000
    # MIN_LIDAR_QUALITY = 5
    # OUTLIER_NB_NEIGHBORS = 10
    # OUTLIER_STD_RATIO = 2.5
    # DUPLICATE_VOXEL_SIZE = 20.0
    # DYNAMIC_DISTANCE_THRESHOLD = 150.0
    # ROBOT_AXIS_LENGTH_MM = 300 

    scanner = LidarScanner(port=LIDAR_PORT)
    if not scanner.connect():
        logging.error("Không thể kết nối LiDAR, thoát chương trình.")
        print("Lỗi: Không thể kết nối LiDAR, thoát chương trình.")
        exit(1)
    
    scanner.start()
    time.sleep(0.1) 
    
    global_map = o3d.geometry.PointCloud()
    global_pose = np.eye(4)
    last_slam_time = time.time()
    prev_points_global = None
    current_points_global = np.array([])  
    
    occupancy_map = np.full((MAP_DIM_PIXELS, MAP_DIM_PIXELS, 3), 150, dtype=np.uint8)
    map_center_px = (MAP_DIM_PIXELS // 2, MAP_DIM_PIXELS // 2)
    
    logging.info("Bắt đầu quét và lập bản đồ thời gian thực.")
    print("\nBắt đầu quét và lập bản đồ thời gian thực. Nhấn 'q' trên cửa sổ để thoát.")
    
    cv2.namedWindow("Real-time SLAM Map", cv2.WINDOW_NORMAL)

    try:
        while scanner.is_initialized():
            scan_data = scanner.get_scan()
            if scan_data is None or len(scan_data) == 0:
                logging.warning("Không nhận được dữ liệu quét, thử lại sau 0.2s.")
                print("Không nhận được dữ liệu quét, thử lại sau 0.2s...")
                time.sleep(0.2)
                continue

            if time.time() - last_slam_time < MIN_SCAN_INTERVAL:
                continue
            
            last_slam_time = time.time()
            current_points = polar_to_cartesian_3d(scan_data)
            
            if len(current_points) < 10:
                logging.warning("Không đủ điểm trong lần quét, bỏ qua.")
                continue
            
            pcd_current = lidar_to_point_cloud(current_points)
            pcd_current = filter_outliers(pcd_current, nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
            current_points = np.asarray(pcd_current.points)
            if len(current_points) < 10:
                logging.warning("Không đủ điểm sau khi lọc, bỏ qua.")
                continue
            
            # Xử lý SLAM
            if len(global_map.points) < 100:
                logging.info("Khởi tạo bản đồ với lần quét đầu tiên...")
                print("Khởi tạo bản đồ với lần quét đầu tiên...")
                global_map.points.extend(o3d.utility.Vector3dVector(current_points))
                current_points_global = current_points
            else:
                map_points_for_icp = np.asarray(global_map.points)
                rmse, transformation_matrix = gicp(current_points, map_points_for_icp, ICP_THRESHOLD, ICP_VOXEL_SIZE, trans_init=global_pose)
                print(f"GICP RMSE: {rmse:.4f}")

                if rmse > 50:
                    logging.warning(f"RMSE cao ({rmse:.4f}), pose không được cập nhật.")
                    print(f"Cảnh báo: RMSE cao ({rmse:.4f}), pose không được cập nhật.")
                    current_points_global = transform_points(current_points, global_pose[:3, :3], global_pose[:3, 3])
                else:
                    global_pose = transformation_matrix
                    current_points_global = transform_points(current_points, global_pose[:3, :3], global_pose[:3, 3])

                    points_to_add = remove_duplicate_points(current_points_global, voxel_size=DUPLICATE_VOXEL_SIZE)
                    points_to_add = remove_dynamic_points(points_to_add, prev_points_global, DYNAMIC_DISTANCE_THRESHOLD)
                    if hasattr(update_occupancy_map, "occupancy_probs"):
                        points_to_add = filter_new_points_by_occupancy(
                            points_to_add,
                            update_occupancy_map.occupancy_probs,
                            map_center_px,
                            RESOLUTION_MM_PER_PIXEL
                        )

                    if len(points_to_add) > 0:
                        global_map.points.extend(o3d.utility.Vector3dVector(points_to_add))
                        logging.info(f"Đã thêm {len(points_to_add)} điểm tĩnh vào bản đồ.")
                
                    if len(global_map.points) > 1000:
                        global_map = downsample_point_cloud(global_map, ICP_VOXEL_SIZE / 2.0)
                        logging.info(f"Giảm mẫu bản đồ, số điểm hiện tại: {len(global_map.points)}")

            prev_points_global = current_points_global.copy()
            robot_pos_map = global_pose[:3, 3]
            update_occupancy_map(occupancy_map, current_points_global, robot_pos_map, map_center_px, RESOLUTION_MM_PER_PIXEL)
            if hasattr(update_occupancy_map, "occupancy_probs"):
                        num_points_before = len(global_map.points)
                        global_map = prune_global_map(
                            global_map,
                            update_occupancy_map.occupancy_probs,
                            map_center_px,
                            RESOLUTION_MM_PER_PIXEL
                        )
            map_for_display = occupancy_map.copy()
            scan_on_map(map_for_display, current_points_global, map_center_px, RESOLUTION_MM_PER_PIXEL, color=(0, 255, 0))
            draw_robot_pose(map_for_display, global_pose, map_center_px, RESOLUTION_MM_PER_PIXEL, ROBOT_AXIS_LENGTH_MM)

            cv2.imshow("Real-time SLAM Map", map_for_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        logging.info("Người dùng dừng chương trình bằng Ctrl+C.")
        print("Người dùng dừng chương trình bằng Ctrl+C.")
    
    finally:
        if len(global_map.points) > 0:
            print("\nĐang xử lý và lưu bản đồ cuối cùng...")
            final_map = downsample_point_cloud(global_map, voxel_size=ICP_VOXEL_SIZE)
            cv2.imwrite("realtime_occupancy_map.png", occupancy_map)
            logging.info("Đã lưu bản đồ chiếm dụng vào 'realtime_occupancy_map.png'")
            print("✅ Đã lưu bản đồ chiếm dụng vào 'realtime_occupancy_map.png'")
            scanner.stop()
            cv2.destroyAllWindows()   
        else:
            print("\nKhông có đủ dữ liệu để tạo bản đồ.")