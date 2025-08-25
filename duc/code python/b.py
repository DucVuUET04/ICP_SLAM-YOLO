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
from flask import Flask, render_template, Response
import os

# Thiết lập logging để debug
logging.basicConfig(filename='lidar_slam.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Cấu hình Flask ---
app = Flask(__name__)
map_lock = threading.Lock()
latest_map_frame = None # Biến toàn cục để lưu trữ khung hình bản đồ mới nhất

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
                self.lidar.reset()  # Reset LiDAR để đảm bảo trạng thái sạch
                time.sleep(1)
                info = self.lidar.get_info()
                self.lidar.start_motor()
                time.sleep(2)  # Chờ motor ổn định
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
        # Lọc các điểm nằm ngoài cung 270 độ phía trước
        # Giả sử phía trước là 0 độ, ta sẽ lấy góc từ -135 đến 135 độ.
        # Vì RPLidar trả về góc từ 0-360, điều này tương đương với (0 <= góc <= 135) hoặc (225 <= góc < 360)
        is_in_front_arc = (angle <= 135) or (angle >= 225)
        if distance > 0 and distance < 10000 and quality > 5 and is_in_front_arc:
            angle_rad = math.radians(angle)
            x = distance * math.cos(angle_rad)
            y = distance * math.sin(angle_rad)
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

def remove_duplicate_points(points, voxel_size=10.0):
    pcd = lidar_to_point_cloud(points)
    pcd_down = downsample_point_cloud(pcd, voxel_size)
    return np.asarray(pcd_down.points)

def remove_dynamic_points(current_points, prev_points, distance_threshold=100.0):

    if prev_points is None or len(prev_points) == 0:
        return current_points    # Giữ nguyên nếu không có điểm trước

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

def inverse_transform_points(points, rotation_matrix, translation_vector):
    points = np.asarray(points)
    inverse_transformed_points = np.dot(points - translation_vector, rotation_matrix)
    return inverse_transformed_points

def icp(source_points, target_points, threshold, voxel_size, trans_init=np.eye(4)):
    if len(source_points) < 10 or len(target_points) < 10:
        logging.warning("Không đủ điểm để chạy ICP.")
        return float('inf'), np.eye(4)
    
    source_pcd = lidar_to_point_cloud(source_points)
    target_pcd = lidar_to_point_cloud(target_points)
    
    source_pcd_down = downsample_point_cloud(source_pcd, voxel_size)
    target_pcd_down = downsample_point_cloud(target_pcd, voxel_size)
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd_down, target_pcd_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
    
    logging.info(f"ICP completed with RMSE: {reg_p2p.inlier_rmse:.4f}")
    return reg_p2p.inlier_rmse, reg_p2p.transformation

def update_occupancy_map(occupancy_map, points_global, robot_pos, map_center_px, resolution):
    """
    Cập nhật bản đồ chiếm dụng, đảm bảo điểm đen không ghi đè lên vùng trắng đã quét.
    Đường trắng được vẽ đến gần điểm vật cản để không ghi đè pixel cuối.
    
    Args:
        occupancy_map (np.ndarray): Bản đồ chiếm dụng (ảnh RGB, kích thước height x width x 3).
        points_global (np.ndarray): Các điểm 3D toàn cục (N x 3) từ cảm biến (x, y, z).
        robot_pos (np.ndarray): Vị trí robot trong hệ tọa độ toàn cục (x, y, z).
        map_center_px (tuple): Tọa độ pixel của tâm bản đồ (x, y).
        resolution (float): Độ phân giải bản đồ (mm/pixel).
    """
    if len(points_global) == 0:
        return  # Bỏ qua nếu không có điểm
    
    # Tính tọa độ pixel của robot
    robot_x_px = int(map_center_px[0] + robot_pos[0] / resolution)
    robot_y_px = int(map_center_px[1] - robot_pos[1] / resolution)
    
    for point_global in points_global:
        # Tính tọa độ pixel của điểm vật cản
        point_x_px = int(map_center_px[0] + point_global[0] / resolution)
        point_y_px = int(map_center_px[1] - point_global[1] / resolution)
        
        # Kiểm tra tọa độ hợp lệ
        if 0 <= point_x_px < occupancy_map.shape[1] and 0 <= point_y_px < occupancy_map.shape[0]:
            # Tính điểm trung gian cách điểm vật cản 1 pixel
            delta_x = point_x_px - robot_x_px
            delta_y = point_y_px - robot_y_px
            distance = max(1, np.hypot(delta_x, delta_y))  # Khoảng cách Euclidean, tránh chia cho 0
            
            if distance > 2:  # Chỉ vẽ đường nếu khoảng cách đủ lớn
                # Tính điểm trung gian (cách điểm cuối 1 pixel)
                scale = (distance - 2) / distance  # Rút ngắn đường để không chạm điểm cuối
                intermediate_x_px = int(robot_x_px + delta_x * scale)
                intermediate_y_px = int(robot_y_px + delta_y * scale)
                
                # Vẽ đường trắng đến điểm trung gian
                cv2.line(occupancy_map, (robot_x_px, robot_y_px), 
                         (intermediate_x_px, intermediate_y_px), (255, 255, 255), 1)
            
            # Chỉ vẽ điểm đen nếu pixel không phải là màu trắng từ các lần quét trước
            if not np.array_equal(occupancy_map[point_y_px, point_x_px], [255, 255, 255]):
                cv2.circle(occupancy_map, (point_x_px, point_y_px), 1, (0, 0, 0), -1)
        
# --- CÁC ROUTE CỦA FLASK ---

@app.route('/')
def index():
    """Render trang web chính."""
    # render_template sẽ tự động tìm file trong thư mục 'templates'
    return render_template('index.html')

def generate_frames():
    """Hàm generator để stream các khung hình bản đồ một cách ổn định hơn."""
    while True:
        # Giữ lock trong thời gian ngắn nhất có thể để sao chép frame
        with map_lock:
            # Sao chép frame để tránh race condition khi encode
            frame_to_encode = latest_map_frame.copy() if latest_map_frame is not None else None

        # Xử lý sau khi đã nhả lock
        if frame_to_encode is None:
            # Nếu chưa có frame, tạo một frame giữ chỗ để gửi cho client
            # Điều này ngăn trình duyệt bị timeout và cung cấp phản hồi
            placeholder = np.full((800, 800, 3), 200, dtype=np.uint8)
            cv2.putText(placeholder, "Dang cho du lieu ban do...", (50, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            frame_to_encode = placeholder
        
        try:
            # Mã hóa khung hình thành định dạng JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if not flag:
                # Nếu mã hóa thất bại, bỏ qua frame này
                logging.warning("Không thể mã hóa frame thành JPEG.")
                time.sleep(0.1) # Đợi một chút trước khi thử lại
                continue
        except Exception as e:
            # Ghi lại lỗi nếu có bất kỳ lỗi nào xảy ra trong quá trình mã hóa
            logging.error(f"Lỗi khi mã hóa frame: {e}")
            time.sleep(0.1)
            continue

        # Yield khung hình đã mã hóa theo định dạng multipart
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        
        # Điều chỉnh tốc độ stream để không làm quá tải CPU
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    """Route để cung cấp video stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- PHẦN 4: CHƯƠNG TRÌNH CHÍNH KẾT HỢP ---

if __name__ == "__main__":
    # --- Cấu hình ---
    LIDAR_PORT = "COM6"
    ICP_VOXEL_SIZE = 20.0
    ICP_THRESHOLD = 250.0  # Tăng ngưỡng để ổn định
    MIN_SCAN_INTERVAL = 0
    MAP_SIZE_MM = 30000  # Giảm xuống 8m x 8m
    RESOLUTION_MM_PER_PIXEL = 30  # 2cm/pixelS
    MAP_DIM_PIXELS = int(MAP_SIZE_MM / RESOLUTION_MM_PER_PIXEL)
    DUPLICATE_VOXEL_SIZE = 35
    DYNAMIC_DISTANCE_THRESHOLD = 200.0
    
    # --- Khởi tạo ---
    scanner = LidarScanner(port=LIDAR_PORT)
    if not scanner.connect():
        logging.error("Không thể kết nối LiDAR, thoát chương trình.")
        print("Lỗi: Không thể kết nối LiDAR, thoát chương trình.")
        exit(1)
    
    scanner.start()
    time.sleep(0.1)  # Chờ LiDAR khởi tạo
    
    global_map = o3d.geometry.PointCloud()
    global_pose = np.eye(4)
    last_slam_time = time.time()
    prev_points = None
    current_points_global = np.array([])  # Khởi tạo để tránh NameError
    
    occupancy_map = np.full((MAP_DIM_PIXELS, MAP_DIM_PIXELS, 3), 150, dtype=np.uint8)
    map_center_px = (MAP_DIM_PIXELS // 2, MAP_DIM_PIXELS // 2)
    
    logging.info("Bắt đầu quét và lập bản đồ thời gian thực.")
    print("\nBắt đầu quét và lập bản đồ thời gian thực.")
    
    # Khởi chạy Flask trong một luồng riêng biệt
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    print("✅ Web server đã khởi động. Truy cập http://127.0.0.1:5000 để xem bản đồ.")
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
            
            # Lọc ngoại lai
            pcd_current = lidar_to_point_cloud(current_points)
            pcd_current = filter_outliers(pcd_current)
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
                rmse, transformation_matrix = icp(current_points, map_points_for_icp, ICP_THRESHOLD, ICP_VOXEL_SIZE, trans_init=global_pose)
                print(f"ICP RMSE: {rmse:.4f}")
                
                if rmse > 50:
                    logging.warning(f"RMSE cao ({rmse:.4f}), bỏ qua frame.")
                    print(f"Cảnh báo: RMSE cao ({rmse:.4f}), bỏ qua frame.")
                    continue
                
                global_pose = transformation_matrix
                current_points_global = transform_points(current_points, global_pose[:3, :3], global_pose[:3, 3])
                
                # Loại bỏ điểm trùng và điểm động
                current_points_global = remove_duplicate_points(current_points_global, voxel_size=DUPLICATE_VOXEL_SIZE)
                current_points_global = remove_dynamic_points(current_points_global, prev_points, DYNAMIC_DISTANCE_THRESHOLD)
                
                if len(current_points_global) > 0:
                    global_map.points.extend(o3d.utility.Vector3dVector(current_points_global))
                    logging.info(f"Đã thêm {len(current_points_global)} điểm tĩnh vào bản đồ.")
                
                # Giảm mẫu bản đồ nếu quá lớn
                if len(global_map.points) >5000:
                    print(f"diem truoc giam mau: {len(global_map.points)}")
                    global_map = downsample_point_cloud(global_map, ICP_VOXEL_SIZE / 2.0)
                    logging.info(f"Giảm mẫu bản đồ, số điểm hiện tại: {len(global_map.points)}")
                    print(f"diem sau giam mau: {len(global_map.points)}")
            # Cập nhật bản đồ chiếm dụng
            robot_pos_map = global_pose[:3, 3]
            update_occupancy_map(occupancy_map, current_points_global, robot_pos_map, map_center_px, RESOLUTION_MM_PER_PIXEL)
            
            cv2.circle(occupancy_map, (int(map_center_px[0] + robot_pos_map[0] / RESOLUTION_MM_PER_PIXEL),
                                      int(map_center_px[1] - robot_pos_map[1] / RESOLUTION_MM_PER_PIXEL)),
                      3, (0, 0, 255), -1)
            
            # Cập nhật frame cho Flask stream một cách an toàn
            with map_lock:
                latest_map_frame = occupancy_map.copy()
            
    except KeyboardInterrupt:
        logging.info("Người dùng dừng chương trình bằng Ctrl+C.")
        print("Người dùng dừng chương trình bằng Ctrl+C.")
    
    finally:
        if len(global_map.points) > 0:
            print("\nĐang xử lý và lưu bản đồ cuối cùng...")
            final_map = downsample_point_cloud(global_map, voxel_size=ICP_VOXEL_SIZE)
            o3d.io.write_point_cloud("realtime_map.pcd", final_map)
            logging.info("Đã lưu bản đồ điểm vào 'realtime_map.pcd'")
            print("✅ Đã lưu bản đồ điểm vào 'realtime_map.pcd'")
            
            cv2.imwrite("realtime_occupancy_map.png", occupancy_map)
            logging.info("Đã lưu bản đồ chiếm dụng vào 'realtime_occupancy_map.png'")
            print("✅ Đã lưu bản đồ chiếm dụng vào 'realtime_occupancy_map.png'")
        else:
            print("\nKhông có đủ dữ liệu để tạo bản đồ.")
        
        scanner.stop()
        # Không cần cv2.destroyAllWindows() nữa