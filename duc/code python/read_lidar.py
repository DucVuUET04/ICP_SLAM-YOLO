import numpy as np
from rplidar import RPLidar, RPLidarException
import time
import serial.tools.list_ports
import threading
import cv2
import math
import os # Thêm thư viện os để quản lý đường dẫn và tên tệp

 
 
def get_com_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]
 
def check_com_port(port_name):
    return port_name in get_com_ports()
 
class LidarScanner:

    def __init__(self, port="COM6", baudrate=256000):
        self.port = port
        self.baudrate = baudrate
        self.lidar = None
        self.latest_scan = None
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

    def connect(self):
        if not check_com_port(self.port):
            print(f"Lỗi: Không tìm thấy cổng {self.port}. Các cổng có sẵn: {get_com_ports()}")
            return False
        try:
            print(f"Đang kết nối tới LiDAR trên cổng {self.port}...")
            self.lidar = RPLidar(self.port, self.baudrate)
            print(f"✅ Kết nối LiDAR thành công. Thông tin: {self.lidar.get_info()}")
            return True
        except RPLidarException as e:
            print(f"Lỗi khi kết nối LiDAR: {e}")
            self.lidar = None
            return False

    def start(self):
        if self.lidar is None and not self.connect():
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("🌀 Bắt đầu luồng đọc dữ liệu LiDAR.")

    def stop(self):

        print("🔴 Đang dừng LiDAR...")
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()  
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        print("✅ LiDAR đã dừng.")

    def _read_loop(self):

        try:
            for scan in self.lidar.iter_scans():
                if not self._running:
                    break
                with self._lock:
                    self.latest_scan = np.array(scan)
        except RPLidarException as e:
            print(f"Lỗi trong luồng đọc LiDAR: {e}")

    def get_scan(self):
        with self._lock:
            return self.latest_scan.copy() if self.latest_scan is not None else None
 

if __name__ == "__main__":

    IMG_HEIGHT = 800
    IMG_WIDTH = 800
    MAX_DISTANCE = 3000.0

    img_radius = IMG_WIDTH / 2.0
    SCALE_FACTOR = MAX_DISTANCE / img_radius
    scan_save_counter = 0
    SAVE_INTERVAL = 0.1
    
    SAVE_DIRECTORY = "scan_data_1" 
   
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    last_save_time = time.time() 

    scanner = LidarScanner(port="COM6", baudrate=256000)
    scanner.start()

    try:
        while True:
            image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
            
            cv2.circle(image, (center_x, center_y), 5, (150, 150, 150), -1)

      
            scan_data = scanner.get_scan()
     
            if scan_data is not None and len(scan_data) > 0:
                for point in scan_data:
                
                    angle = point[1]
                    distance = point[2]

                    if distance == 0:
                        continue

                    px = (distance / SCALE_FACTOR) * math.cos(math.radians(angle))
                    py = (distance / SCALE_FACTOR) * math.sin(math.radians(angle))
             
                    point_on_image = (int(center_x + px), int(center_y + py))
                    cv2.circle(image, point_on_image, 1, (0, 255, 0), -1)

            cv2.imshow("Real-time Lidar Scan", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

           
            if time.time() - last_save_time >= SAVE_INTERVAL:
                print(f"\n--- Tự động lưu sau {SAVE_INTERVAL} giây ---")
                if scan_data is not None and len(scan_data) > 0:
                    scan_save_counter += 1
                
                    raw_scan_filename = os.path.join(SAVE_DIRECTORY, f"scan_data_{scan_save_counter}.npy")
                    np.save(raw_scan_filename, scan_data)
                    print(f"✅ Đã lưu dữ liệu quét thô vào '{raw_scan_filename}'")
                    last_save_time = time.time()
                else:
                    print("Không có dữ liệu để lưu.")
                    last_save_time = time.time()
    finally:

        scanner.stop()
        cv2.destroyAllWindows()