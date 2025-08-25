import open3d as o3d
import numpy as np
import cv2
import time
import os
import threading 
import multiprocessing as mp
import queue 
from flask import Flask, Response, jsonify, render_template,request, send_from_directory,send_file
import webbrowser
import process
import gicp_lidar
import Config
import json
from ultralytics import YOLO
import math
import img

app = Flask(__name__)
map_lock = threading.RLock() 
latest_map_frame = None
latest_icp_frame = None
slam_paused = threading.Event()
display_lock = threading.Event()
show_camera_trigger = mp.Event() 
show_icp = True
model = YOLO(r"C:\Xu_ly_anh\train2\weights\best.pt")
points_of_interest = []
POI_FILE = "points_of_interest.json"
active_target_point_info = None
distance_to_target = None 
# --- PHẦN 1: CẤU HÌNH VÀ TẢI DỮ LIỆU ---

def save_map_data(base_name):
    """
    Lưu bản đồ hiện tại thành 2 file: PNG và NPY từ tên file cơ sở.
    """
    with map_lock:
   
        if map_for_display is None or len(global_map.points) == 0:
            print("Lỗi: Không có dữ liệu bản đồ để lưu.")
            return

        png_filename = base_name + '.png'
        npy_filename = base_name + '.npy'
        
        try:
         
            cv2.imwrite(png_filename, map_for_display)
            print(f"✅ Đã lưu ảnh bản đồ đầy đủ vào {png_filename}")

            all_global_points = np.asarray(global_map.points)
            
          
            resolution = Config.RESOLUTION_MM_PER_PIXEL
            center_px = (Config.MAP_WIDTH_PIXELS // 2, Config.MAP_HEIGHT_PIXELS // 2)
            
            pixel_points = []
            for point in all_global_points:
                px = int(center_px[0] + point[0] / resolution)
                py = int(center_px[1] - point[1] / resolution)
                pixel_points.append([px, py])

            np.save(npy_filename, np.array(pixel_points))
            print(f"✅ Đã lưu toàn bộ {len(pixel_points)} điểm bản đồ vào {npy_filename}")

        except Exception as e:
            error_message = f"Lỗi nghiêm trọng khi lưu bản đồ: {str(e)}"
            print(f"❌ {error_message}")

def load_points_of_interest():
    global points_of_interest
    if os.path.exists(POI_FILE):
        try:
            with open(POI_FILE, 'r') as f:
                points_of_interest = json.load(f)
            print(f"✅ Đã tải {len(points_of_interest)} điểm yêu thích từ {POI_FILE}")
        except Exception as e:
            print(f"❌ Lỗi khi tải điểm yêu thích: {e}")

def save_points_of_interest():
    with map_lock:
        try:
            with open(POI_FILE, 'w') as f:
                json.dump(points_of_interest, f, indent=2)
        except Exception as e:
            print(f"❌ Lỗi khi lưu điểm yêu thích: {e}")

def draw_points_of_interest(image, points, map_center_px, resolution):
    for point_mm in points:
        px = int(map_center_px[0] + point_mm[0] / resolution)
        py = int(map_center_px[1] - point_mm[1] / resolution)
        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
            cv2.drawMarker(image, (px, py), color=(0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
def create_icp_visualization(source_points, target_points, robot_pose, resolution):
    """
    Tạo hình ảnh trực quan hóa cho quá trình ICP, được căn giữa tại vị trí của robot.
    - source_points (màu xanh): Dữ liệu quét hiện tại (trong hệ tọa độ cục bộ của robot).
    - target_points (màu đỏ): Các điểm từ bản đồ (trong hệ tọa độ toàn cục) được chuyển về hệ tọa độ của robot.
    """
    icp_map = np.full((Config.IMG_SIZE, Config.IMG_SIZE, 3), 128, dtype=np.uint8)
    icp_map_center_px = (Config.IMG_SIZE // 2, Config.IMG_SIZE // 2)

    # Chuyển các điểm của bản đồ mục tiêu (target_points) từ tọa độ toàn cục (global)
    # về tọa độ cục bộ (local) của robot để so sánh.
    try:
        inv_pose = np.linalg.inv(robot_pose)
        target_points_local = gicp_lidar.transform_points(target_points, inv_pose[:3, :3], inv_pose[:3, 3])
    except np.linalg.LinAlgError:
        # Nếu ma trận pose không khả nghịch, không thể tạo hình ảnh
        cv2.putText(icp_map, "Error: Pose not invertible", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return icp_map

    # Vẽ các điểm mục tiêu (đã ở local frame) lên bản đồ ICP - màu đỏ
    for point in target_points_local:
        if np.isnan(point[0]) or np.isnan(point[1]):
            continue
        px = int(icp_map_center_px[0] + point[0] / resolution)
        py = int(icp_map_center_px[1] - point[1] / resolution)
        if 0 <= px < icp_map.shape[1] and 0 <= py < icp_map.shape[0]:
            cv2.circle(icp_map, (px, py), 1, (0, 0, 255), 1) # Màu đỏ cho bản đồ

    # Vẽ các điểm quét hiện tại (source_points, vốn đã ở local frame) - màu xanh lá
    for point in source_points:
        if np.isnan(point[0]) or np.isnan(point[1]):
            continue
        px = int(icp_map_center_px[0] + point[0] / resolution)
        py = int(icp_map_center_px[1] - point[1] / resolution)
        if 0 <= px < icp_map.shape[1] and 0 <= py < icp_map.shape[0]:
            cv2.circle(icp_map, (px, py), 1, (0, 255, 0), 1) # Màu xanh cho lần quét hiện tại

    return icp_map

# --- Tiến trình xử lý camera (Process) ---
def camera_process_worker(frame_queue, trigger_event, stop_event, save_dir):
    """
    Tiến trình này xử lý tất cả các tương tác với camera để không làm treo tiến trình chính.
    """
    cap = None
 
    print("[Camera Process] Bắt đầu...")

    while not stop_event.is_set():
     
        is_triggered = trigger_event.wait(timeout=0.1)

        if is_triggered:
            if cap is None:
                print("[Camera Process] Kích hoạt. Đang thử khởi tạo camera...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap2= cv2.VideoCapture(2, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("[Camera Process] LỖI: Không thể mở camera. Sẽ thử lại sau 1 giây.")
                    cap = None
                    time.sleep(1)
                    continue
                else:
                    print("[Camera Process] ✅ Camera đã được khởi tạo thành công.")

          
            if cap is not None:
                ret, frame = cap.read()
                ret2, frame2 = cap2.read()
                if ret and ret2:
                    img_1 = frame.copy()
                    img_2 = frame2.copy()
                    img_gray = np.full((420, 640), 150, dtype=np.uint8)

                    try:
                        # Gửi một tuple chứa cả hai khung hình đã xử lý
                        frame_queue.put_nowait((img_1, img_2))
                    except queue.Full:
                        pass  

                    results_1 = model.predict(img_1, task="detect", save=False, conf=0.5, verbose=False)
                    results_2 = model.predict(img_2, task="detect", save=False, conf=0.5, verbose=False)

                    points_right_corners = None
                    points_left_corners = None

                    r = results_1[0]
                    boxes = r.boxes  
                    if len(boxes) > 0:
                        box = boxes[0]  
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        points_right_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        cv2.polylines(img_1, [points_right_corners], isClosed=True, color=(0, 255, 0), thickness=2)
                        cv2.line(img_1,(320,0),(320,480),(0,255,0),5)
                    r2 = results_2[0]
                    boxes2 = r2.boxes  

                    if len(boxes2) > 0:
                        box2 = boxes2[0]  
                        x1_2, y1_2, x2_2, y2_2 = box2.xyxy[0].cpu().numpy().astype(int)
                        conf = box2.conf[0].item()
                        center_x_pallet=(x1_2+x2_2)//2
                        center_y_pallet=(y1_2+y2_2)//2
                        h,w,_ = img_2.shape
                        delta_x= center_x_pallet-w/2
                        delta_x_pixel= x2_2-x1_2
                    
                    
                        cls = int(box2.cls[0].item())
                        points_left_corners = np.array([[x1_2, y1_2], [x2_2, y1_2], [x2_2, y2_2], [x1_2, y2_2]], dtype=np.int32)
                        cv2.polylines(img_2, [points_left_corners], isClosed=True, color=(0, 255, 0), thickness=2)
                        cv2.line(img_2,(w//2,0),(w//2,h),(0,0,255),2)
                        cv2.circle(img_2,(int(center_x_pallet),int(center_y_pallet)),3,(0,0,255),-1)
                
                    if points_left_corners is not None and points_right_corners is not None:
                        try:
                            corners_3d = img.stereo_to_3d(points_left_corners, points_right_corners, Config.f, Config.cx, Config.cy, Config.B)
                            normal, angle_rad, mean_depth = img.pallet_orientation_and_distance(corners_3d)
                            angle_deg = -np.degrees(angle_rad)
                            pallet_center_3d = np.mean(corners_3d, axis=0)
                            angle_horizontal_rad = np.arctan2(pallet_center_3d[0], pallet_center_3d[2])
                            angle_horizontal_deg = np.degrees(angle_horizontal_rad)
                            distance_x =13-mean_depth * math.tan(angle_horizontal_rad)
                            x_pallet_mm= 110 
                        
                            palet_img_x=x_pallet_mm* math.cos(angle_rad)
                            px_mm= delta_x_pixel/palet_img_x

                            dis_lech= (delta_x / px_mm)-13
                            cv2.putText(img_2,str(dis_lech),(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                        
                            print("---")
                            print("Pháp tuyến pallet:", normal)
                            print(f"Góc nghiêng (so với trục Z): {angle_deg:.2f}°")  
                            print(f"Góc lệch ngang (so với tâm camera): {(angle_horizontal_deg):.2f}°")
                            print(f"Khoảng cách (theo Z): {mean_depth:.2f} mm")
                            print(f"Khoảng cách (theo X): {distance_x:.2f} mm")
                            if (180-angle_horizontal_deg) > 5:  
                                chieu_lech= 'Lech phai'
                                print("==> Chiều lệch: Lệch Phải")
                            elif (180-angle_horizontal_deg) < -5:
                                chieu_lech= 'Lech trai'
                                print("==> Chiều lệch: Lệch Trái")
                            else:
                                chieu_lech= 'Chinh giua'
                                print("==> Chiều lệch: Chính Giữa")
                            cv2.putText(img_gray,f"Goc nghieng Pallet: {angle_deg:.2f}deg",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                            cv2.putText(img_gray,f"Goc lech camera: {(angle_horizontal_deg):.2f}deg",(20,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                            cv2.putText(img_gray,f"Khoang cach toi Pallet: {mean_depth:.2f} mm",(20,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                            cv2.putText(img_gray,f"Do lech camera: {dis_lech:.2f} mm",(20,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                            cv2.putText(img_gray,f"Chieu lech: {chieu_lech}",(20,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                            cv2.imshow("result",img_gray)
                        except Exception as e:
                            print(f"Lỗi tính toán 3D: {e}")

        else:
           
            if cap is not None:
                print("[Camera Process] Tắt. Đang giải phóng tài nguyên camera.")
                cap.release()
                cap = None
                try:
                   
                    frame_queue.put_nowait((None, None))
                except queue.Full:
                    pass

    if cap is not None:
        cap.release()
    print("[Camera Process] Đã dừng.")
def update_map():
  
    # Biến đếm để thực hiện bảo trì bản đồ định kỳ
    map_maintenance_counter = 0

    global global_map, global_pose, prev_points_global, map_for_display, current_points_global, latest_map_frame, latest_icp_frame,map_for_display_2, distance_to_target
    for i in range(Config.START_FILE + 1, Config.END_FILE):
        try:
            if slam_paused.is_set():
                time.sleep(0.5)
                continue
            scan_file = Config.BASE_PATH.format(i)
            scan_data = process.load_and_prepare_scan(scan_file) 

            if scan_data is None or len(scan_data) == 0:
                print(f"Lỗi tải tệp {scan_file} hoặc tệp rỗng. Bỏ qua.")
                continue

            current_points = scan_data

            if len(current_points) < 10:
                continue

            pcd_current = gicp_lidar.lidar_to_point_cloud(current_points)
            pcd_current = process.filter_outliers(pcd_current, nb_neighbors=Config.OUTLIER_NB_NEIGHBORS, std_ratio=Config.OUTLIER_STD_RATIO)
            current_points = np.asarray(pcd_current.points)
            if len(current_points) < 10:
                continue
            

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
                
          
            rmse, transformation_matrix =gicp_lidar.gicp(current_points,map_points_for_icp, Config.ICP_THRESHOLD, Config.GICP_VOXEL_SIZE, trans_init=global_pose)
        
            print(f"RMSE: {rmse:.4f}")


            if rmse <= Config.MAX_RMSE_THRESHOLD:
                global_pose = transformation_matrix
                current_points_global = gicp_lidar.transform_points(current_points, global_pose[:3, :3], global_pose[:3, 3])
                
                points_to_add = process.remove_duplicate_points(current_points_global, voxel_size=Config.DUPLICATE_VOXEL_SIZE)
                points_to_add = process.remove_dynamic_points(points_to_add, prev_points_global, Config.DYNAMIC_DISTANCE_THRESHOLD)
                if hasattr(process.update_occupancy_map, "occupancy_probs"):
                        points_to_add = process.filter_new_points_by_occupancy(
                            points_to_add, 
                            process.update_occupancy_map.occupancy_probs,
                            map_center_px,
                            Config.RESOLUTION_MM_PER_PIXEL 
                        )

                if len(points_to_add) > 0:
                    global_map.points.extend(o3d.utility.Vector3dVector(points_to_add))
            else:
                print(f"Cảnh báo: RMSE cao ({rmse:.4f}) tại tệp {scan_file}, pose không được cập nhật.")
                   
            prev_points_global = current_points_global.copy()
            robot_pos_map = global_pose[:3, 3]

            points_for_occupancy = process.remove_duplicate_points(current_points_global, voxel_size=Config.RESOLUTION_MM_PER_PIXEL * 2.0)

            process.update_occupancy_map(map_for_display, points_for_occupancy, robot_pos_map, map_center_px, Config.RESOLUTION_MM_PER_PIXEL)
            
            map_maintenance_counter += 1
            
            if map_maintenance_counter >= Config.MAP_MAINTENANCE_INTERVAL:
                map_maintenance_counter = 0
                
                if hasattr(process.update_occupancy_map, "occupancy_probs"):
                    num_points_before = len(global_map.points)
                    global_map = process.prune_global_map(
                        global_map,
                        process.update_occupancy_map.occupancy_probs,
                        map_center_px,
                        Config.RESOLUTION_MM_PER_PIXEL
                    )
                    if len(global_map.points) < num_points_before:
                       
                        pass
                if len(global_map.points) > 1000:
                    num_points_before = len(global_map.points)
                    global_map = gicp_lidar.downsample_point_cloud(global_map, Config.ICP_VOXEL_SIZE)
                   
                
            map_for_display_2 = map_for_display.copy()
            
       
            if active_target_point_info:
                target_point_mm = active_target_point_info['pos_mm']
                process.draw_target_point(map_for_display_2, target_point_mm, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, color=(0, 255, 0))
                
                distance = process.calculate_distance(robot_pos_map[:2], target_point_mm)
                distance_to_target = distance 
                
                cv2.putText(map_for_display_2, f"Distance to Target: {distance:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

               
                if distance <= Config.CAMERA_TRIGGER_DISTANCE_MM: 
                    show_camera_trigger.set() # Bật cờ
                else:
                    show_camera_trigger.clear() # Tắt cờ
            else:
                distance_to_target = None 
                show_camera_trigger.clear() 

            process.scan_on_map(map_for_display_2, current_points_global, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, color=(0, 255, 0))
            process.draw_robot_pose(map_for_display_2, global_pose, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, Config.ROBOT_AXIS_LENGTH_MM)
            icp_frame = create_icp_visualization(current_points, map_points_for_icp, global_pose, Config.RESOLUTION_MM_PER_PIXEL)

            if not display_lock.is_set():
                with map_lock:
                    latest_map_frame = map_for_display_2.copy()
                    latest_icp_frame = icp_frame.copy()
        except Exception as e:
            print(f"!!!!!!!!!! LỖI KHÔNG MONG MUỐN TRONG VÒNG LẶP SLAM !!!!!!!!!")
            print(f"Lỗi xảy ra khi xử lý file: {scan_file}")
            print(f"Chi tiết lỗi: {e}")
            import traceback
            traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
   


# --- CÁC ROUTE CỦA FLASK ---
@app.route('/load_map_for_imshow', methods=['POST'])
def load_map_for_imshow():
    global latest_map_frame, latest_icp_frame
    data = request.get_json()
    filename = data.get('filename')

    if not filename or not os.path.exists(filename):
        
        return jsonify({"status": "error", "message": "File không tồn tại."}), 404

    try:
     
        display_lock.set()
       
      
        loaded_image = cv2.imread(filename)
        if loaded_image is None:
             return jsonify({"status": "error", "message": "Không thể đọc file ảnh."}), 500

        with map_lock: 
            # latest_map_frame = loaded_image
            latest_icp_frame = np.full((loaded_image.shape[0], loaded_image.shape[1], 3), 128, dtype=np.uint8)
            cv2.putText(latest_icp_frame, "Saved Map View", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        return jsonify({"status": "success", "message": f"Đã hiển thị bản đồ {filename} trên server."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi server: {e}"}), 500


@app.route('/save_map')
def save_map():
    """
    Nhận yêu cầu lưu bản đồ từ giao diện web.
    """
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"status": "error", "message": "Lỗi: Tên file không được cung cấp."}), 400

    base_name = os.path.splitext(filename)[0]
    
    if not base_name:
         return jsonify({"status": "error", "message": "Lỗi: Tên file không hợp lệ."}), 400

    try:
        save_map_data(base_name)
        message = f"Bản đồ đã được lưu thành công với tên '{base_name}'"
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        error_message = f"Lỗi phía server khi lưu bản đồ: {str(e)}"
        return jsonify({"status": "error", "message": error_message}), 500
@app.route('/list_saved_files')
def list_saved_files():
    try:
    
        files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.png')]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/add_point', methods=['POST'])
def add_point_of_interest():
    global points_of_interest
    with map_lock:
        if global_pose is None:
            return jsonify({"status": "error", "message": "Vị trí robot chưa xác định."}), 400
        
        robot_pos_mm = global_pose[:2, 3].tolist() 
        
        points_of_interest.append(robot_pos_mm)
        print(f"📍 Đã thêm điểm yêu thích mới tại: {robot_pos_mm}")
        
        save_points_of_interest()

    return jsonify({"status": "success", "message": "Đã thêm điểm thành công.", "new_point": robot_pos_mm})

@app.route('/set_active_target', methods=['POST'])
def set_active_target():
    global active_target_point_info
    data = request.get_json()
    point_id = data.get('id')

    if point_id is None:
        active_target_point_info = None
        print("🎯 Đã hủy mục tiêu đang hoạt động.")
        return jsonify({"status": "success", "message": "Đã hủy mục tiêu."})

    with map_lock:
        try:
            point_id = int(point_id)
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "ID điểm không hợp lệ."}), 400

        if 0 <= point_id < len(points_of_interest):
            pos_mm = points_of_interest[point_id]
            active_target_point_info = {"id": point_id, "pos_mm": pos_mm}
            print(f"🎯 Đã đặt mục tiêu hoạt động là Điểm {point_id + 1} tại {pos_mm}")
            return jsonify({"status": "success", "message": f"Đã đặt mục tiêu là Điểm {point_id + 1}"})
        else:
            active_target_point_info = None
            return jsonify({"status": "error", "message": "ID điểm không tồn tại."}), 400

@app.route('/get_points_of_interest')
def get_points_of_interest():
    resolution = Config.RESOLUTION_MM_PER_PIXEL
    center_px = (Config.MAP_WIDTH_PIXELS // 2, Config.MAP_HEIGHT_PIXELS // 2)
    
    points_data = []
    with map_lock:
        for i, point_mm in enumerate(points_of_interest):
            px = int(center_px[0] + point_mm[0] / resolution)
            py = int(center_px[1] - point_mm[1] / resolution)
            points_data.append({
                "id": i,
                "name": f"Điểm {i + 1}",
                "pos_px": (px, py)
            })
            
    return jsonify({"points": points_data})
@app.route('/get_map_points/<filename_base>')
def get_map_points(filename_base):
    """
    Tải file .npy chứa các điểm, chuyển thành JSON và gửi cho client.
    """
    npy_filename = filename_base + '.npy'
    try:
        points = np.load(npy_filename)
        points_list = points.tolist()
        if len(points_list) > 0:
            points_list = points_list[:-1]
        return jsonify({"points": points_list})
        
    except FileNotFoundError:
        return jsonify({"points": []})
    except Exception as e:
        return jsonify({"error": "Không thể tải dữ liệu điểm"}), 500

@app.route('/get_map_image/<path:filename>')
def get_map_image(filename):
    print(f"\n[SERVER LOG] ➡️ Nhận yêu cầu cho ảnh: {filename}")
    try:
        image_path = os.path.join(os.getcwd(), filename)
        print(f"[SERVER LOG] ℹ️ Đường dẫn đầy đủ của ảnh: {image_path}")

        if not os.path.exists(image_path):
            print(f"[SERVER LOG] ❌ LỖI: Không tìm thấy file ảnh tại đường dẫn trên!")
            return "File not found", 404
        

        print(f"[SERVER LOG] ✅ Đã tìm thấy ảnh, đang đọc file vào bộ nhớ...")
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"[SERVER LOG] ✅ Đã đọc xong, đang gửi {len(image_bytes)} bytes đi...")
        return Response(image_bytes, mimetype='image/png')

    except Exception as e:
        print(f"[SERVER LOG] ❌ LỖI NGHIÊM TRỌNG trong get_map_image: {e}")
        return jsonify({'error': f'Lỗi server khi đọc ảnh: {str(e)}'}), 500
@app.route('/')
def index():
    return render_template('jjj.html')

def generate_frames():
    global is_first_frame, current_points_global

    while True:
        with map_lock:
            points = current_points_global.copy() if current_points_global is not None else None

        if points is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
                b'Content-Type: application/json\r\n\r\n' +
                bytearray(jsonify({"points": points.tolist()}).data) + b'\r\n')

        time.sleep(0.05)

@app.route("/map_image")
def map_image():
    with map_lock:
        map_frame = None#latest_map_frame.copy() if latest_map_frame is not None else None

    if map_frame is None:
        map_frame = np.full((Config.IMG_SIZE, Config.IMG_SIZE, 3), 0, dtype=np.uint8)
    

    _, encoded_img = cv2.imencode(".jpg", map_frame)
    return Response(encoded_img.tobytes(), mimetype="image/jpeg")
@app.route("/points_stream")
def points_stream():
    def generate():
        resolution = Config.RESOLUTION_MM_PER_PIXEL
        center_px = (Config.MAP_WIDTH_PIXELS // 2, Config.MAP_HEIGHT_PIXELS // 2)
        
        while True:
            points_global, current_pose, dist = None, None, None
            with map_lock:
                # Thay đổi: Gửi toàn bộ điểm trong global_map thay vì chỉ lần quét hiện tại
                if global_map is not None and len(global_map.points) > 0:
                    points_global = current_points_global.copy()
                if global_pose is not None:
                    current_pose = global_pose.copy()
                if distance_to_target is not None:
                    dist = distance_to_target

            payload = {}
            if points_global is not None and len(points_global) > 0:
                points_in_pixels = []
                for point in points_global:
                    px = int(center_px[0] + point[0] / resolution)
                    py = int(center_px[1] - point[1] / resolution)
                    points_in_pixels.append((px, py))
                payload['points'] = points_in_pixels

            if current_pose is not None:
                robot_pos_mm = current_pose[:3, 3]
                robot_rot_mat = current_pose[:3, :3]
                robot_x_px = int(center_px[0] + robot_pos_mm[0] / resolution)
                robot_y_px = int(center_px[1] - robot_pos_mm[1] / resolution)
                
                axis_vec = np.array([Config.ROBOT_AXIS_LENGTH_MM, 0, 0])
                end_vec_mm = (robot_rot_mat @ axis_vec)
                end_x_px = int(robot_x_px + end_vec_mm[0] / resolution)
                end_y_px = int(robot_y_px - end_vec_mm[1] / resolution)
                payload['pose'] = {'x': robot_x_px, 'y': robot_y_px, 'ex': end_x_px, 'ey': end_y_px}

            if dist is not None:
                payload['distance'] = f"{dist:.2f}"
            
            if payload:
                yield f"data: {json.dumps(payload)}\n\n"
            
            time.sleep(0.1)  
            
    return Response(generate(), mimetype="text/event-stream")
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/toggle_visibility', methods=['POST'])
def toggle_visibility():
    global show_map, show_icp
    data = request.get_json()
    show_map = data.get('map', show_map)
    show_icp = data.get('icp', show_icp)
    return jsonify(status="success", show_map=show_map, show_icp=show_icp)

@app.route('/stop_stream')
def stop_stream():
    slam_paused.set()
    return jsonify(status="success", message="Stream stopped")

@app.route('/resume_stream')
def resume_stream():
    display_lock.clear() 
    slam_paused.clear()
    return jsonify(status="success", message="Stream resumed")

@app.route('/save_frame')
def save_frame():
    global latest_map_frame, latest_icp_frame
    with map_lock:
        if latest_map_frame is not None and latest_icp_frame is not None:
   
            icp_frame_resized = cv2.resize(latest_icp_frame, (latest_map_frame.shape[1], latest_map_frame.shape[0]))
            combined_frame = cv2.addWeighted(latest_map_frame, 0.5, icp_frame_resized, 0.5, 0)
            filename = f"capture_{int(time.time())}.png"
            cv2.imwrite(filename, combined_frame)
            return jsonify(status="success", filename=filename)
            
    return jsonify(status="error", message="No frame to save"), 404

@app.route('/load_map/<filename>')
def load_map(filename):
    global occupancy_map, global_map, update_mode

    filepath = os.path.join(Config.SAVE_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"message": f"File {filename} không tồn tại"}), 404

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        occupancy_map = cv2.imread(filepath)
    elif filename.lower().endswith('.pcd'):
        global_map = o3d.io.read_point_cloud(filepath)
    else:
        return jsonify({"message": "Định dạng file không hỗ trợ"}), 400

    update_mode = 0  
    return jsonify({"message": f"Đã tải bản đồ {filename} và chuyển sang chế độ định vị"})
@app.route('/capture_map')
def capture_map():
    global capture_only
    capture_only = True
    return jsonify({"message": "Đang chụp ảnh bản đồ..."})



if __name__ == "__main__":
    model = YOLO(r"C:\Xu_ly_anh\train2\weights\best.pt")

    load_points_of_interest()
    occupancy_map = np.full((Config.MAP_HEIGHT_PIXELS, Config.MAP_WIDTH_PIXELS, 3), 128, dtype=np.uint8)
    map_center_px = (Config.MAP_WIDTH_PIXELS // 2, Config.MAP_HEIGHT_PIXELS // 2)
    global_map = o3d.geometry.PointCloud()
    global_pose = np.eye(4)
    prev_points_global = None
    current_points_global = np.array([])
    map_for_display_2 = None 

    if not os.path.exists(Config.save_dir):
        os.makedirs(Config.save_dir)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    camera_frame_queue = mp.Queue(maxsize=2)
    stop_camera_process = mp.Event()
    camera_process = mp.Process(target=camera_process_worker, 
                                args=(camera_frame_queue, show_camera_trigger, stop_camera_process, Config.save_dir),
                                daemon=True)
    camera_process.start()
    print("\nStarting data processing and mapping. Press 'q' or ESC to exit.")

    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    url = "http://127.0.0.1:5000"
    print(f"✅ Web server started. Opening {url} in browser...")
    time.sleep(0.1)
    webbrowser.open_new_tab(url)
  
    slam_thread = threading.Thread(target=update_map, daemon=True)
    
    try:
       
        first_scan_points = process.load_and_prepare_scan(Config.BASE_PATH.format(Config.START_FILE))
        if first_scan_points is None or len(first_scan_points) == 0:
            print("Error: Could not load the first scan. Exiting.")
            exit()
        
        global_map.points.extend(o3d.utility.Vector3dVector(first_scan_points))
        current_points_global = first_scan_points
        process.update_occupancy_map(occupancy_map, current_points_global, global_pose[:3, 3], map_center_px, Config.RESOLUTION_MM_PER_PIXEL)
        
        map_for_display = occupancy_map.copy() 
        map_for_display_2= map_for_display.copy()
        process.draw_robot_pose(map_for_display, global_pose, map_center_px, Config.RESOLUTION_MM_PER_PIXEL, Config.ROBOT_AXIS_LENGTH_MM)
        icp_frame = np.full((Config.IMG_SIZE, Config.IMG_SIZE, 3), 128, dtype=np.uint8)
        
        with map_lock:
            # latest_map_frame = map_for_display.copy()
            latest_icp_frame = icp_frame.copy()
            
        slam_thread.start()

        while True:
            map_to_show = None
            icp_to_show = None
            with map_lock:
                map_to_show = latest_map_frame.copy() if latest_map_frame is not None else None
                icp_to_show = latest_icp_frame.copy() if latest_icp_frame is not None else None
            
            if map_to_show is not None:
                cv2.imshow("Real-time SLAM Map", map_to_show)
            
            try:
                cam_right, cam_left = camera_frame_queue.get_nowait()
                if cam_right is not None and cam_left is not None:
                    cv2.imshow("Right Camera", cam_right)
                    cv2.imshow("Left Camera", cam_left)
                else:
                    # Đóng cửa sổ camera nếu không có frame
                    if cv2.getWindowProperty("Right Camera", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Right Camera")
                    if cv2.getWindowProperty("Left Camera", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Left Camera")
            except queue.Empty:
              
                pass

            if icp_to_show is not None: 
                icp_to_show = cv2.resize(icp_to_show, (800, 800))

                cv2.imshow("ICP Visualization", icp_to_show)

            key = cv2.waitKey(Config.DELAY_MS)
            if key in [ord('q'), 27]:
                break

    except KeyboardInterrupt:
        print("User stopped the program with Ctrl+C.")

    finally:
        print("\nĐang dừng chương trình...")
       
        stop_camera_process.set()
        if camera_process is not None:
           
            while not camera_frame_queue.empty():
                try: camera_frame_queue.get_nowait()
                except queue.Empty: break
            camera_process.join(timeout=2) 

        if len(global_map.points) > 0:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\nNot enough data to create a map.")
