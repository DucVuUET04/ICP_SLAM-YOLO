import cv2 
import numpy as np
import open3d as o3d
import math
import os
import time
def lidar_to_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud
def downsample_point_cloud(point_cloud, voxel_size):
    if not point_cloud.has_points():
        return point_cloud
    return point_cloud.voxel_down_sample(voxel_size)

def perform_icp(source_pcd, target_pcd, threshold, trans_init=np.identity(4)):
    """
    Thực hiện đăng ký ICP giữa hai đám mây điểm.
    """
    print(":: Bắt đầu đăng ký ICP (Iterative Closest Point)...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    return reg_p2p.transformation, reg_p2p.inlier_rmse

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

def draw_points(image, points, color, center_x, center_y, scale, size=2):
    """Hàm tiện ích để vẽ các điểm lên ảnh."""
    for point in points:
        px = int(center_x + point[0] / scale)
        py = int(center_y + point[1] / scale)
        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
            cv2.circle(image, (px, py), size, color, -1)

if __name__ == "__main__":
    # --- CẤU HÌNH ---
    scan_file_target = r"Scan_data_1\scan_data_350.npy"  # Đám mây điểm mục tiêu (cố định)
    scan_file_source = r"Scan_data_1\scan_data_355.npy"  # Đám mây điểm nguồn (sẽ được di chuyển)

    IMG_HEIGHT, IMG_WIDTH = 800, 800 
    MAX_DISTANCE = 9000.0  # Khoảng cách tối đa để hiển thị (mm), khớp với bộ lọc trong polar_to_cartesian_3d
    VOXEL_SIZE = 50.0      # Kích thước voxel để giảm mẫu (tăng tốc ICP)
    ICP_THRESHOLD = 200.0  # Ngưỡng khoảng cách cho ICP

    # --- BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU ---
    target_points = load_and_prepare_scan(scan_file_target)
    source_points = load_and_prepare_scan(scan_file_source)
    
    if target_points is None or source_points is None or len(target_points) == 0 or len(source_points) == 0:
        print(f"Lỗi tải tệp hoặc tệp rỗng. Kết thúc chương trình.")
        exit()
        
    print(f"Số điểm Target: {len(target_points)} | Số điểm Source: {len(source_points)}")

    # Chuyển đổi sang Open3D PointCloud
    pcd_target = lidar_to_point_cloud(target_points)
    pcd_source = lidar_to_point_cloud(source_points)

    # Giảm mẫu để tăng tốc ICP
    pcd_target_down = downsample_point_cloud(pcd_target, voxel_size=VOXEL_SIZE)
    pcd_source_down = downsample_point_cloud(pcd_source, voxel_size=VOXEL_SIZE)

    # --- BƯỚC 2: THỰC HIỆN ICP ---
    transformation_matrix, rmse = perform_icp(pcd_source_down, pcd_target_down, ICP_THRESHOLD)
    print(f"ICP hoàn tất. RMSE: {rmse:.4f}")
    print("Ma trận biến đổi tìm được:")
    print(transformation_matrix)

    # Áp dụng phép biến đổi lên đám mây điểm nguồn GỐC (để trực quan hóa chất lượng cao)
    pcd_source_aligned = pcd_source.transform(transformation_matrix)
    source_points_aligned = np.asarray(pcd_source_aligned.points)

    # --- BƯỚC 3: TRỰC QUAN HÓA BẰNG CV2 ---
    # Tạo một ảnh duy nhất để hiển thị kết quả sau ICP
    display_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    
    center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
    SCALE_FACTOR = MAX_DISTANCE / (IMG_WIDTH / 2.0)

    # Vẽ các đám mây điểm để so sánh
    draw_points(display_image, target_points, (0, 255, 0), center_x, center_y, SCALE_FACTOR,size=1)  
    draw_points(display_image, source_points, (0, 0, 255), center_x, center_y, SCALE_FACTOR, size=1)
    draw_points(display_image, source_points_aligned, (255, 0, 0), center_x, center_y, SCALE_FACTOR,size=1) 

    # Thêm nhãn chú thích cho ảnh
    cv2.putText(display_image, "ICP Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_image, "Target: Green | Original Source: Red", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_image, "Aligned Source: Blue", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Vòng lặp chính để hiển thị hình ảnh
    while True:
        # Hiển thị hình ảnh
        cv2.imshow("ICP Result", display_image)
        
        # Chờ người dùng nhấn phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
