import cv2
import json
import os
import math, shutil
import numpy as np
import time
from tkinter.messagebox import showerror, showwarning, showinfo
from lib_main import edit_csv_tab, remove
from ultralytics import YOLO

# chuyen dia chi co dau \ sang /
def edit_path(input):
    new_path = ""
    for i in list(input):
        if i == str("\\"):
            new_path = new_path + "/"
        if i != str("\\"):
            new_path = new_path + i
    return new_path
path_phan_mem = edit_path(os.path.dirname(os.path.realpath(__file__)))
if path_phan_mem.split("/")[-1] == "_internal":
    path_phan_mem = path_phan_mem.replace("/_internal","")

# Load YOLO model for auto-segmentation
try:
    # Make sure to provide the correct path to your trained model
    model = YOLO(r"C:\20210327\labels_segmentation\test\yolo11_seg_singleclass_2\weights\best.pt")
    model_loaded = True
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Could not load YOLO model: {e}. Auto-segmentation will be unavailable.")
    model_loaded = False


# Initialize global variables
drawing = False
current_polygon = []
# polygons = [] # Thay đổi này
polygons_data_list = [] # Biến mới để lưu trữ danh sách các dictionary
polygons_cv = []
polygons_op = []
start_x, start_y = 0, 0
center = (0, 0)
number_resize = 5
anpha = 1
state_file = 'current_state.txt'
edit = 0
label = None # Biến label này sẽ tạm thời cho việc lựa chọn nhãn khi vẽ thủ công, hoặc có thể được dùng khi bạn muốn gán nhãn cho một contour cụ thể
name_label = {"0": ["giot dau 1",[]], "1": ["giot dau 2",[]], "2": ["none",[]], "3": ["none",[]], "4": ["none",[]],
                "5": ["none",[]], "6": ["none",[]], "7": ["none",[]], "8": ["none",[]], "9": ["none",[]]}
mouse_pos = (-1, -1)
eraser_size = 20
is_adding = False
is_erasing = False
move_mask_mode = False
is_moving_mask = False
selected_polygon_index = -1
# Thêm các biến để di chuyển mask một cách ổn định
original_polygon_for_move = None
initial_click_pos = None

eraser_shape = 'square'
show_help = True

# Thêm biến trạng thái hiển thị bounding box và danh sách bounding box YOLO
show_bbox = False
yolo_bboxes = []  # Lưu bounding box từ YOLO predict


# Thêm biến cho chế độ vẽ bounding box thủ công
manual_bbox_mode = False
manual_bbox_start = None
manual_bbox_end = None
manual_bbox_list = []  # Lưu các bbox thủ công [(x1, y1, x2, y2), ...]

def run_yolo_on_bbox(image, bbox, model):
    # bbox: (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    results = model.predict(
        source=crop,
        imgsz=1024,
        conf=0.25,
        show=False,
        verbose=False
    )
    masks = []
    for r in results:
        if r.masks is not None:
            for mask_tensor in r.masks.data:
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
    return masks

# Đọc trạng thái show_bbox từ file state nếu có
def load_bbox_state(state_file):
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('show_bbox='):
                        return line.strip().split('=')[1] == 'True'
    except Exception:
        pass
    return False
show_bbox = load_bbox_state(state_file)

text = {
    "s": "tu dong phan doan",
    "d": "anh tiep theo",
    "a": "quay lai",
    "g": "xoay phai",
    "h": "xoay trai",
    "e": "chinh sua",
    "r": "Reset",
    "z": "xoa nhan",
    "u": "cap nhat",
    "c": "Copy (Slot 1)",
    "v": "Paste (Slot 1)",
    "o": "Copy (Slot 2)",
    "b": "Paste (Slot 2)",
    "t": "di chuyen mask",
    "p": "bat/tat bounding box",
    "m": "ve bounding box thu cong",
    "Chuot phai": "xoa",
    "Chuot trai": "them hoac dich chuyen",
    "Chuot giua": "loai tay",
    "cuon chuot": "kich thuoc tay",
    "q": "Quit & Save State",
    "?": "Show/Hide Help"
}

def read_settings(file_path):
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, value = line.strip().split(' ')
            settings[str(name)] = str(value)
            print(settings[str(name)])
    return settings
path_setting = path_phan_mem + "/setting/setting_segmentation.txt"
data_setting = read_settings(path_setting)

def create_help_image(text_dict, width=450, height=500):
    """Creates an image with help text."""
    help_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    y_pos = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 25

    # Title
    cv2.putText(help_img, "Hotkeys & Controls", (20, y_pos), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    y_pos += 40

    for key, description in text_dict.items():
        line = f"- {key}: {description}"
        cv2.putText(help_img, line, (20, y_pos), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        y_pos += line_height
        if y_pos > height - 20:
            # In a real scenario, might need multiple columns or a larger image
            break
    return help_img

def get_screen_dimensions():
    """Lấy chiều rộng và chiều cao của màn hình."""
    try:
        # Sử dụng Tkinter để lấy kích thước màn hình một cách an toàn
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        # Giá trị dự phòng nếu Tkinter không khả dụng
        return 1920, 1080

folder_input = edit_path(data_setting["folder_input"])
if data_setting["forder_output"] == "None":
    forder_output = 'labels_segmentation/output'
else:
    forder_output = edit_path(data_setting["forder_output"])
number_resize = float(data_setting["resize"])
anpha = int(float(data_setting["anpha"]))
# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(forder_output):
    os.makedirs(forder_output)
# Function to calculate the center of a polygon
def calculate_center(points):
    if len(points) == 0:
        return None
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return int(center_x), int(center_y)

# Function to rotate a point around a center
def rotate_point(point, center, angle):
    angle_rad = math.radians(angle)
    x, y = point
    cx, cy = center
    x_new = cx + math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy)
    y_new = cy + math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy)
    return int(x_new), int(y_new)

# Function to save polygons to a TXT file
def save_polygons_to_txt(img, file_path, polygons_data_list): # Thêm tham số label
    # polygons_data_list.append({"label": assigned_label, "polygon": current_poly})
    remove.remove_file(file_path)
    if len(polygons_data_list) > 0:
        if len(img.shape) == 3:
            img_height, img_width, _ = img.shape
        else:
            img_height, img_width = img.shape
        data = ""
        with open(file_path, 'w') as f:
            for polygon_data in polygons_data_list: # Đổi tên biến để tránh trùng lặp với tham số polygons
                label = polygon_data["label"]
                list_data = polygon_data["polygon"]
                data = data + label
                for poly in list_data:
                    x = poly[0] / img_width
                    y = poly[1] / img_height
                    # Đảm bảo các giá trị nằm trong khoảng [0, 1]
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    data = data + " " + str(x) + " " + str(y)
                data = data + "\n"
            f.write(f"{data}")

def load_polygons_from_txt(file_path, img_width, img_height):
    global name_label
    all_polygons_data = []  # Thay đổi để lưu trữ danh sách các dictionary
    if not os.path.exists(file_path):
        return [] # Return empty list if file does not exist
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label_loaded = parts[0]
            points_data = parts[1:]

            poly = []
            for i in range(0, len(points_data), 2):
                x = float(points_data[i]) * img_width
                y = float(points_data[i + 1]) * img_height
                poly.append((int(x), int(y)))

            # Tạo một dictionary cho mỗi polygon và thêm vào danh sách
            polygon_entry = {"label": str(label_loaded), "polygon": poly}
            # Dòng dưới đây là nguyên nhân gây lỗi: nó lưu lại tâm của nhãn cũ, khiến chúng bị gán lại sau khi xóa.
            # center = calculate_center(poly)
            # name_label[str(label_loaded)][1].append(center)
            all_polygons_data.append(polygon_entry)

    return all_polygons_data

def auto_segment(image_to_predict, model):
    """
    Runs YOLO model to get segmentation masks and updates the global img_mask.
    This will overwrite any existing masks on the image.
    """
    global img_mask, polygons_data_list, yolo_bboxes

    # Clear existing data
    polygons_data_list.clear()
    img_mask.fill(0)
    yolo_bboxes.clear()

    # Predict
    results = model.predict(
        source=image_to_predict.copy(),  # Use a copy to avoid any potential modification by predict
        imgsz=1024,
        conf=0.25,  # Lower confidence to catch more, user can delete false positives
        show=False,
        verbose=False
    )

    h, w = image_to_predict.shape[:2]

    for r in results:
        # Lưu bounding box nếu có
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                # box: [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)
                yolo_bboxes.append((x1, y1, x2, y2))

        if r.masks is None:
            continue
        for mask_tensor in r.masks.data:
            # Convert mask to numpy
            mask = mask_tensor.cpu().numpy()
            # Resize mask to the working image size
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Add the mask to our global mask. Use bitwise_or to combine masks.
            img_mask[mask > 0.5] = 255
# Function to save the current state
def save_current_state(file_path, current_image):
    # Lưu cả trạng thái show_bbox
    with open(file_path, 'w') as f:
        f.write(f"{current_image}\n")
        f.write(f"show_bbox={show_bbox}\n")
# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
# Function to load the current state
def load_current_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                # Lấy dòng đầu tiên là tên ảnh
                return lines[0].strip()
    return None
# Mouse callback function

def draw_polygon(event, x, y, flags, param):
    global drawing, current_polygon, start_x, start_y, edit, name_label
    global label, mouse_pos, is_adding, eraser_size, eraser_shape, is_erasing, img_mask, polygons_data_list
    global manual_bbox_mode, manual_bbox_start, manual_bbox_end, manual_bbox_list
    global move_mask_mode, is_moving_mask, selected_polygon_index, original_polygon_for_move, initial_click_pos


    mouse_pos = [x, y]

    # Chế độ di chuyển mask (Ưu tiên cao nhất)
    if move_mask_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_polygon_index = -1
            for i in range(len(polygons_data_list) - 1, -1, -1):
                entry = polygons_data_list[i]
                contour_poly = np.array(entry["polygon"], dtype=np.int32)
                if len(contour_poly) > 0 and cv2.pointPolygonTest(contour_poly, (x, y), False) >= 0:
                    selected_polygon_index = i
                    is_moving_mask = True
                    # --- Logic mới: Lưu trạng thái ban đầu ---
                    initial_click_pos = (x, y)
                    # Lưu một bản sao của polygon gốc để tính toán di chuyển
                    original_polygon_for_move = list(entry["polygon"])
                    break  # Đã tìm thấy, thoát vòng lặp

        elif event == cv2.EVENT_MOUSEMOVE:
            if is_moving_mask and selected_polygon_index != -1:
                # --- Logic mới: Tính toán dựa trên vị trí ban đầu ---
                dx = x - initial_click_pos[0]
                dy = y - initial_click_pos[1]

                poly_entry = polygons_data_list[selected_polygon_index]
                
                # Tạo polygon mới bằng cách dịch chuyển polygon GỐC
                new_poly = []
                for px, py in original_polygon_for_move:
                    new_poly.append((px + dx, py + dy))
                poly_entry["polygon"] = new_poly

        elif event == cv2.EVENT_LBUTTONUP:
            if is_moving_mask:
                img_mask.fill(0)
                for entry in polygons_data_list:
                    poly = np.array(entry["polygon"], dtype=np.int32)
                    if len(poly) > 0:
                        cv2.fillPoly(img_mask, [poly], 255)
            # Reset lại tất cả trạng thái di chuyển
            is_moving_mask = False
            selected_polygon_index = -1
            original_polygon_for_move = None
            initial_click_pos = None

        return

    # Chế độ vẽ bounding box thủ công
    if manual_bbox_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            manual_bbox_start = (x, y)
            manual_bbox_end = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            manual_bbox_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            manual_bbox_end = (x, y)
            drawing = False
            # Lưu bbox vừa vẽ vào danh sách
            x1, y1 = manual_bbox_start
            x2, y2 = manual_bbox_end
            x1, x2 = sorted([max(0, x1), max(0, x2)])
            y1, y2 = sorted([max(0, y1), max(0, y2)])
            if x2-x1 > 10 and y2-y1 > 10:
                manual_bbox_list.append((x1, y1, x2, y2))
        return

    if label is not None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if mouse_pos is within a contour
            found_contour = False
            for entry_idx, entry in enumerate(polygons_data_list):
                contour_poly = np.array(entry["polygon"], dtype=np.int32)
                if len(contour_poly) > 0 and cv2.pointPolygonTest(contour_poly, (x, y), False) >= 0:
                    polygons_data_list[entry_idx]["label"] = str(label)
                    found_contour = True
                    break # Assign label to the first contour found and exit

            if not found_contour:
                # If clicked outside any existing contour, add a new point for the selected label if desired.
                # Currently, this part of the code adds points to a fixed label's sample points for drawing new polygons.
                # This might need re-evaluation based on desired manual drawing behavior.
                # For now, let's keep the original behavior for manual label point addition IF no contour is clicked.
                name_label[str(label)][1].append(mouse_pos)


            label_width = 90
            label_height = (h2 - 110) // 10
            offset_x = 10
            offset_y = 10

            for i in range(10):
                x1 = w2 + offset_x
                y1 = offset_y + i * (label_height + offset_y)
                x2 = w2 + label_width + offset_x
                y2 = offset_y + (i + 1) * (label_height + offset_y)

                if x > x1 and x < x2 and y > y1 and y < y2:
                    if label == i:
                        label = None
                    else:
                        label = i # Gán label bằng số thứ tự của ô vuông
    else:
        if event == cv2.EVENT_MBUTTONDOWN:
            if eraser_shape == 'square':
                eraser_shape = 'circle'
            else:
                eraser_shape = 'square'
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Lăn lên để tăng kích thước
                eraser_size += 2
            else:  # Lăn xuống để giảm kích thước
                eraser_size -= 2
            eraser_size = max(1, min(100, eraser_size))  # Giới hạn kích thước trong khoảng 1-100
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            is_adding = False
            is_erasing = True
            start_x, start_y = x, y
        if event == cv2.EVENT_LBUTTONUP:
            is_adding = False
            drawing = False
        elif event == cv2.EVENT_RBUTTONUP:
            is_erasing = False
            drawing = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            is_erasing = False
            is_adding = True

            label_width = 90
            label_height = (h2 - 110) // 10
            offset_x = 10
            offset_y = 10

            for i in range(10):
                x1 = w2 + offset_x
                y1 = offset_y + i * (label_height + offset_y)
                x2 = w2 + label_width + offset_x
                y2 = offset_y + (i + 1) * (label_height + offset_y)

                if x > x1 and x < x2 and y > y1 and y < y2:
                    if label == i:
                        label = None
                    else:
                        label = i # Gán label bằng số thứ tự của ô vuông
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Di chuyển các polygon hiện có trong polygons_data_list
                for pol_data_entry in polygons_data_list:
                    # Check if the mouse is within this polygon before moving it
                    contour_poly = np.array(pol_data_entry["polygon"], dtype=np.int32)
                    if len(contour_poly) > 0 and cv2.pointPolygonTest(contour_poly, (start_x, start_y), False) >= 0:
                        dx = x - start_x
                        dy = y - start_y
                        poly = []
                        for px, py in pol_data_entry["polygon"]:
                            px += dx
                            py += dy
                            poly.append([px, py])
                        pol_data_entry["polygon"] = poly
                        break # Move only the first polygon found under the mouse click
                start_x, start_y = x, y


        if is_adding and img_mask is not None: # them
            print("thêm")
            x1 = x - eraser_size // 2
            y1 = y - eraser_size // 2
            x2 = x + eraser_size // 2
            y2 = y + eraser_size // 2
            if eraser_shape == 'square':
                cv2.rectangle(img_mask, (x1, y1), (x2, y2), 255, -1)
            else:
                cv2.circle(img_mask, mouse_pos, eraser_size // 2, 255, -1)

        elif is_erasing and img_mask is not None: # xoa
            print("xóa")
            x1 = x - eraser_size // 2
            y1 = y - eraser_size // 2
            x2 = x + eraser_size // 2
            y2 = y + eraser_size // 2
            if eraser_shape == 'square':
                cv2.rectangle(img_mask, (x1, y1), (x2, y2), 0, -1)
            else:
                cv2.circle(img_mask, mouse_pos, eraser_size // 2, 0, -1)


list_name_img = os.listdir(folder_input)
if len(list_name_img) == 0:
    print("khong co anh")
stt = 0
name_img_old = ""

last_checked_image = load_current_state(state_file)
if last_checked_image and last_checked_image in list_name_img:
    stt = list_name_img.index(last_checked_image)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_polygon)
img_mask = None
img = None
img_new = None
img_temp = None
img_resize = None

# Variable to hold a message to be displayed on the image
display_message = ""
message_start_time = 0
MESSAGE_DURATION_MS = 2000 # 2 seconds

# Lấy kích thước màn hình một lần khi bắt đầu
screen_width, screen_height = get_screen_dimensions()

while True:
    name_img = list_name_img[stt]
    # if name_img_old != name_img or ((stt == 0 or stt == len(list_name_img) - 1) and len(list_name_img) != 0):


    if name_img_old != name_img and len(list_name_img) != 0:
        # Xóa các điểm mẫu của ảnh trước để không ảnh hưởng đến ảnh hiện tại
        for key in name_label:
            name_label[key][1] = []

        output_file = os.path.join(forder_output, name_img.split(".")[0] + ".txt")
        img = cv2.imread(os.path.join(folder_input, name_img))
        if img is None:
            print(f"Could not load image {os.path.join(folder_input, name_img)}. Skipping.")
            stt = (stt + 1) % len(list_name_img) # Move to next image
            continue

        h, w, _ = img.shape
        img_resize = cv2.resize(img.copy(), (int(w * number_resize), int(h * number_resize)))
        # polygons_data_list
        h2, w2, _ = img_resize.shape
        img_mask = np.zeros((h2,w2),dtype=np.uint8)
        print("img_mask 2")
        img_new = np.ones((h2,w2+100,3), dtype=np.uint8) * 255
        img_temp = img_new.copy()
        img_new[:h2 ,:w2 ,:] = img_resize.copy()
        polygons_data_list = load_polygons_from_txt(output_file, w2, h2) # Tải dữ liệu vào biến mới

        # Nếu có dữ liệu cũ, vẽ lại mask từ đó
        if polygons_data_list:
            for entry in polygons_data_list:
                poly = np.array(entry["polygon"], dtype=np.int32)
                if len(poly) > 0: # Ensure polygon is not empty
                    cv2.fillPoly(img_mask, [poly], 255) # Vẽ lại các polygon lên mask

        # Sau khi đã cập nhật img_resize, mới lấy bounding box YOLO nếu show_bbox đang bật
        yolo_bboxes.clear()
        if show_bbox and model_loaded:
            results = model.predict(
                source=img_resize.copy(),
                imgsz=1024,
                conf=0.25,
                show=False,
                verbose=False
            )
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        yolo_bboxes.append((x1, y1, x2, y2))

        name_img_old = name_img

    if img_new is not None and img_mask is not None:
        img_new = np.ones((h2,w2+100,3), dtype=np.uint8) * 255
        img_new[:h2 ,:w2 ,:] = img_resize.copy()
        # Vẽ bounding box thủ công khi đang vẽ
        if manual_bbox_mode:
            # Vẽ bbox đang kéo
            if manual_bbox_start and manual_bbox_end:
                x1, y1 = manual_bbox_start
                x2, y2 = manual_bbox_end
                cv2.rectangle(img_new, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Vẽ tất cả bbox đã vẽ
            for bbox in manual_bbox_list:
                cv2.rectangle(img_new, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
    else:
        img_mask = np.zeros((500,500),dtype=np.uint8)
        print("zeros")
        img_resize = np.zeros((500,500,3),dtype=np.uint8)
        img_new = np.zeros((500,500+100,3),dtype=np.uint8) # Ensure it has enough width

    # Chỉ cập nhật danh sách polygon từ mask nếu không ở trong chế độ di chuyển.
    # Khi đang di chuyển, polygons_data_list là nguồn dữ liệu chính xác.
    if not is_moving_mask:
        # Tìm các đường viền trên mask hiện tại
        contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Logic mới để duy trì nhãn một cách ổn định ---
        updated_polygons_data_list = []
        
        # Đối với mỗi đường viền mới tìm thấy trong khung hình này...
        for new_contour in contours:
            new_poly_list = [tuple(point[0]) for point in new_contour]
            new_center = calculate_center(new_poly_list)
            
            best_match_label = "none"
            min_dist = float('inf')
            
            # ...hãy tìm đường viền cũ gần nhất từ khung hình trước.
            if new_center:
                for old_entry in polygons_data_list:
                    if old_entry['label'] != 'none':
                        old_center = calculate_center(old_entry['polygon'])
                        if old_center:
                            is_inside = cv2.pointPolygonTest(np.array(old_entry['polygon'], dtype=np.int32), new_center, False) >= 0
                            if is_inside:
                                dist = calculate_distance(new_center, old_center)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match_label = old_entry['label']

            assigned_label = best_match_label
            
            if assigned_label == "none":
                for label_key, (name, points_for_label) in name_label.items():
                    if label_key != "none":
                        for pt in points_for_label:
                            if cv2.pointPolygonTest(new_contour, (pt[0], pt[1]), False) >= 0:
                                assigned_label = label_key
                                break
                    if assigned_label != "none":
                        break

            updated_polygons_data_list.append({"label": assigned_label, "polygon": new_poly_list})

        # Cập nhật danh sách chính bằng danh sách các đa giác đã được xử lý
        polygons_data_list = updated_polygons_data_list

    # Ensure all sample points in name_label are still within some contour
    # This part is tricky. A simple approach is to clear and repopulate based on current contours.
    # However, the user might manually place sample points for future manual drawing.
    # For now, let's assume sample points are mainly for initial contour assignment,
    # and direct labeling by clicking on a contour handles re-labeling.
    # If the intent is for sample points to "stick" to polygons, a more robust point-to-polygon tracking is needed.
    # For this request, we prioritize checking 'none' labels in polygons_data_list.

    img_new = img_new[:h2,:w2+100]

    # Tạo một lớp phủ (overlay) để vẽ các vùng màu bán trong suốt
    overlay = img_new.copy()
    alpha = 0.2  # Độ trong suốt

    # Màu sắc cho các nhãn, bạn có thể tùy chỉnh
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0)]
    
    # --- Bước 1: Vẽ tất cả các vùng đa giác bán trong suốt lên lớp phủ ---
    for entry in polygons_data_list:
        poly_to_draw = np.array(entry["polygon"], dtype=np.int32)
        if len(poly_to_draw) == 0:
            continue
        
        label_str = entry.get("label", "none")
        
        # Nếu polygon đã có nhãn (là một số)
        if label_str.isdigit():
            label_index = int(label_str)
            if 0 <= label_index < len(colors):
                color = colors[label_index]
                cv2.fillPoly(overlay, [poly_to_draw], color)
        elif label_str == "none":
            # Vẽ các vùng chưa gán nhãn bằng màu vàng để dễ nhận biết
            unlabeled_color = (0, 255, 255)  # Màu vàng (B, G, R)
            cv2.fillPoly(overlay, [poly_to_draw], unlabeled_color)
    # --- Bước 2: Trộn lớp phủ vào ảnh chính ---
    cv2.addWeighted(overlay, alpha, img_new, 1 - alpha, 0, img_new)


    # --- Bước 3: Vẽ các chi tiết không trong suốt (tâm và nhãn) lên ảnh đã trộn ---
    for entry in polygons_data_list:
        center = calculate_center(entry["polygon"])
        if center:
            # Vẽ điểm trung tâm cho TẤT CẢ các vùng (giống logic cũ)
            cv2.circle(img_new, center, 5, (255, 255, 255), -1) # Chấm trắng
            cv2.circle(img_new, center, 6, (0, 0, 0), 1) # Viền đen cho dễ nhìn

    # --- Chỉ vẽ bounding box do YOLO tạo ra nếu show_bbox ---
    if show_bbox:
        for bbox in yolo_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_new, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Màu xanh lá cây

    # Vẽ các khung vuông và label ở bên phải ảnh
    label_width = 90
    label_height = (h2 - 110) // 10
    offset_x = 10
    offset_y = 10
    for i in range(10):
        x1 = w2 + offset_x
        y1 = offset_y + i * (label_height + offset_y)
        x2 = w2 + label_width + offset_x
        y2 = offset_y + (i + 1) * (label_height + offset_y)

        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0)]
        color = colors[i]

        label_text = name_label[str(i)][0]
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = int((x2 + x1 - text_size[0]) / 2)
        text_y = int((y2 + y1 + text_size[1]) / 2)
        cv2.rectangle(img_new, (x1, y1), (x2, y2), color, -1)
        cv2.putText(img_new, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if label == i:
            if int(time.time() * 1000) % 500 > 250:
                cv2.rectangle(img_new, (x1, y1), (x2, y2), (0, 0, 0), 3)

    if mouse_pos != (-1, -1):
        color = (255, 255, 255) if not is_adding else (0, 255, 255)
        if eraser_shape == 'square':
            x_start = mouse_pos[0] - eraser_size // 2
            y_start = mouse_pos[1] - eraser_size // 2
            x_end = mouse_pos[0] + eraser_size // 2
            y_end = mouse_pos[1] + eraser_size // 2
            cv2.rectangle(img_new, (x_start, y_start), (x_end, y_end), color, 2)
        else:
            cv2.circle(img_new, mouse_pos, eraser_size // 2, color, 2)

    # Display temporary message if any
    if display_message and (cv2.getTickCount() - message_start_time) / cv2.getTickFrequency() * 1000 < MESSAGE_DURATION_MS:
        cv2.putText(img_new, display_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # showerror(title="error", message=display_message)
    else:
        display_message = "" # Clear message after duration

    # Hiển thị các cửa sổ
    cv2.imshow(name_img, img_new)
    if show_help:
        help_image = create_help_image(text, height=h2)
        cv2.imshow('Help', help_image)

    # --- Căn chỉnh vị trí các cửa sổ ra giữa màn hình ---
    if 'w2' in locals() and 'h2' in locals():
        main_width = img_new.shape[1]
        main_height = img_new.shape[0]

        if show_help:
            help_width = 450  # Chiều rộng cố định từ create_help_image
            total_width = help_width + main_width  # Không còn khoảng cách giữa 2 cửa sổ
            start_x = max(0, (screen_width - total_width) // 2)
            start_y = max(0, (screen_height - main_height) // 2)

            cv2.moveWindow('Help', start_x, start_y) # Cửa sổ hotkeys bên trái
            cv2.moveWindow(name_img, start_x + help_width, start_y) # Cửa sổ ảnh bên phải
        else:
            # Chỉ căn giữa cửa sổ chính nếu không hiển thị trợ giúp
            start_x = max(0, (screen_width - main_width) // 2)
            start_y = max(0, (screen_height - main_height) // 2)
            cv2.moveWindow(name_img, start_x, start_y)

    cv2.setMouseCallback(name_img, draw_polygon)


    k = cv2.waitKey(1) & 0xFF

    # Phím tắt bật/tắt chế độ di chuyển mask
    if k == ord('t'):
        move_mask_mode = not move_mask_mode
        if move_mask_mode:
            display_message = "Che do di chuyen mask: BAT. Nhan giu chuot de keo."
            # Tắt các chế độ khác có thể xung đột
            manual_bbox_mode = False
            label = None
            is_adding = False
            is_erasing = False
            drawing = False
        else:
            display_message = "Che do di chuyen mask: TAT."
            is_moving_mask = False # Đảm bảo thoát khỏi trạng thái đang di chuyển
        message_start_time = cv2.getTickCount()


    # Phím tắt bật/tắt chế độ vẽ bounding box thủ công
    elif k == ord('m'):
        if not manual_bbox_mode:
            manual_bbox_mode = True
            manual_bbox_start = None
            manual_bbox_end = None
            manual_bbox_list = []
            display_message = "Ve bounding box thu cong: keo chuot trai de ve, bam m lan nua de ket thuc"
            message_start_time = cv2.getTickCount()
        else:
            # Khi tắt chế độ, thực hiện YOLO mask cho tất cả bbox đã vẽ
            if len(manual_bbox_list) > 0:
                # Lấy YOLO predict toàn ảnh nếu chưa có
                if not hasattr(draw_polygon, '_yolo_results') or draw_polygon._yolo_img_id != id(img_resize):
                    draw_polygon._yolo_results = model.predict(
                        source=img_resize.copy(),
                        imgsz=1024,
                        conf=0.25,
                        show=False,
                        verbose=False
                    )
                    draw_polygon._yolo_img_id = id(img_resize)
                results = draw_polygon._yolo_results
                def bbox_iou(boxA, boxB):
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[2], boxB[2])
                    yB = min(boxA[3], boxB[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                    return iou
                threshold = 0.3
                mask_count = 0
                for bbox in manual_bbox_list:
                    found = False
                    for r in results:
                        if hasattr(r, 'boxes') and r.boxes is not None and r.masks is not None:
                            for idx, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                                x1b, y1b, x2b, y2b = map(int, box)
                                iou = bbox_iou(bbox, (x1b, y1b, x2b, y2b))
                                if iou > threshold:
                                    mask = r.masks.data[idx].cpu().numpy()
                                    mask = cv2.resize(mask, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST)
                                    mask_bin = (mask > 0.5).astype(np.uint8) * 255
                                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    for contour in contours:
                                        if len(contour) > 2:
                                            contour_shifted = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                                            polygons_data_list.append({"label": "none", "polygon": contour_shifted})
                                            cv2.fillPoly(img_mask, [np.array(contour_shifted, dtype=np.int32)], 255)
                                    mask_count += 1
                                    found = True
                                    break
                        if found:
                            break
                if mask_count > 0:
                    print(f"Đã vẽ {mask_count} mask tự động cho các bounding box trùng với YOLO!")
                else:
                    print("Không có bounding box YOLO nào trùng với các vùng bạn chọn!")
            manual_bbox_mode = False
            manual_bbox_start = None
            manual_bbox_end = None
            manual_bbox_list = []

    # Check for unlabeled polygons before saving or navigating
    unlabeled_polygons_exist = False
    for entry in polygons_data_list:
        if entry["label"] == "none":
            unlabeled_polygons_exist = True
            break

    if k == ord('p'):
        # Nếu chưa có bounding box YOLO cho ảnh hiện tại thì chạy YOLO để lấy bounding box (không tạo mask)
        if not yolo_bboxes and model_loaded:
            # Chạy YOLO chỉ để lấy bounding box, không tạo mask
            yolo_bboxes.clear()
            results = model.predict(
                source=img_resize.copy(),
                imgsz=1024,
                conf=0.25,
                show=False,
                verbose=False
            )
            h, w = img_resize.shape[:2]
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        yolo_bboxes.append((x1, y1, x2, y2))
        show_bbox = not show_bbox
        # Hiển thị thông báo trạng thái
        display_message = f"Bounding box: {'ON' if show_bbox else 'OFF'}"
        message_start_time = cv2.getTickCount()

    if k == ord('q') or cv2.getWindowProperty(name_img, cv2.WND_PROP_VISIBLE) < 1:
        if unlabeled_polygons_exist:
            display_message = "Vui lòng gán nhãn cho tất cả các vùng hình bao!"
            message_start_time = cv2.getTickCount()
            continue # Prevent quitting
        save_current_state(state_file, name_img)
        output_file = os.path.join(forder_output, name_img.split(".")[0] + ".txt")
        save_polygons_to_txt(img_mask, output_file, polygons_data_list)
        break
    elif k == ord('?'):
        show_help = not show_help
        if not show_help:
            try:
                cv2.destroyWindow('Help')
            except cv2.error:
                pass
    elif k == ord('s'):
        if model_loaded:
            # Display a message to the user
            temp_display = img_new.copy()
            cv2.putText(temp_display, "Running Auto-Segmentation...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(name_img, temp_display)
            cv2.waitKey(1) # Allow window to update

            auto_segment(img_resize, model)

            # Sau khi tạo mask, cập nhật luôn bounding box YOLO cho ảnh hiện tại và bật hiển thị
            yolo_bboxes.clear()
            results = model.predict(
                source=img_resize.copy(),
                imgsz=1024,
                conf=0.25,
                show=False,
                verbose=False
            )
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        yolo_bboxes.append((x1, y1, x2, y2))
            show_bbox = True

            display_message = "Auto-segmentation complete. Please label new masks."
            message_start_time = cv2.getTickCount()
        else:
            display_message = "YOLO model not loaded. Auto-segmentation unavailable."
            message_start_time = cv2.getTickCount()
    elif k == ord('g'):
        # Áp dụng xoay cho tất cả các polygon
        for entry in polygons_data_list:
            if entry["polygon"]:
                center_poly = calculate_center(entry["polygon"])
                if center_poly:
                    entry["polygon"] = [rotate_point(p, center_poly, anpha) for p in entry["polygon"]]
    elif k == ord('h'):
        for entry in polygons_data_list:
            if entry["polygon"]:
                center_poly = calculate_center(entry["polygon"])
                if center_poly:
                    entry["polygon"] = [rotate_point(p, center_poly, -1 * anpha) for p in entry["polygon"]]
    elif k == ord('d'):
        if unlabeled_polygons_exist:
            display_message = "Vui lòng gán nhãn cho tất cả các vùng hình bao!"
            message_start_time = cv2.getTickCount()
            continue # Prevent moving to next image
        cv2.destroyAllWindows()
        if stt < len(list_name_img)-1:
            stt = stt + 1
        else:
            stt = 0  # Quay lại ảnh đầu tiên
        output_file = os.path.join(forder_output, name_img.split(".")[0] + ".txt")
        save_polygons_to_txt(img_mask, output_file, polygons_data_list)
        print(f"Polygons saved to {output_file}")
        #polygons_data_list = [] # Xóa dữ liệu cho ảnh tiếp theo
    elif k == ord('a'):
        if unlabeled_polygons_exist:
            display_message = "Vui lòng gán nhãn cho tất cả các vùng hình bao!"
            message_start_time = cv2.getTickCount()
            continue # Prevent moving to previous image
        cv2.destroyAllWindows()
        if stt > 0:
            stt = stt - 1
        output_file = os.path.join(forder_output, name_img.split(".")[0] + ".txt")
        save_polygons_to_txt(img_mask, output_file, polygons_data_list)
        print(f"Polygons saved to {output_file}")
        #polygons_data_list = []
    elif k == ord('z'):
        if len(polygons_data_list) > 0:
            del polygons_data_list[-1]
            # Cần cập nhật lại img_mask sau khi xóa polygon
            img_mask.fill(0) # Xóa mask hiện tại
            for entry in polygons_data_list:
                poly = np.array(entry["polygon"], dtype=np.int32)
                if len(poly) > 0: # Ensure polygon is not empty
                    cv2.fillPoly(img_mask, [poly], 255)
    elif k == ord('u'): # Clear current polygon points - this might not be relevant anymore as you are getting from contours
        # Logic này cần được xem xét lại nếu bạn không còn "vẽ" polygon thủ công
        # Nếu bạn muốn xóa mask, bạn có thể làm:
        img_mask.fill(0)
        polygons_data_list = []
    elif k == ord('c'): # Copy current polygons to polygons_cv
        polygons_cv = polygons_data_list[:] # Sử dụng slice để tạo bản sao
    elif k == ord('v'): # Paste polygons_cv to polygons
        polygons_data_list = polygons_cv[:]
        # Sau khi dán, cần cập nhật lại img_mask
        img_mask.fill(0)
        for entry in polygons_data_list:
            poly = np.array(entry["polygon"], dtype=np.int32)
            if len(poly) > 0: # Ensure polygon is not empty
                cv2.fillPoly(img_mask, [poly], 255)
    elif k == ord('o'): # Copy current polygons to polygons_op
        polygons_op = polygons_data_list[:]
    elif k == ord('b'): # Paste polygons_op to polygons
        polygons_data_list = polygons_op[:]
        # Sau khi dán, cần cập nhật lại img_mask
        img_mask.fill(0)
        for entry in polygons_data_list:
            poly = np.array(entry["polygon"], dtype=np.int32)
            if len(poly) > 0: # Ensure polygon is not empty
                cv2.fillPoly(img_mask, [poly], 255)
    elif k == ord('e'): # Toggle edit mode - edit mode for what? manual drawing?
        if edit == 0:
            edit = 1
        else:
            edit = 0
    elif k == ord('r'): # Reset/Remove current image's annotations
        polygons_data_list = []
        img_mask.fill(0) # Đảm bảo mask cũng được xóa
        output_file = os.path.join(forder_output, name_img.split(".")[0] + ".txt")
        if os.path.exists(output_file):
            os.remove(output_file)
            print("removed annotations for current image")

cv2.destroyAllWindows()