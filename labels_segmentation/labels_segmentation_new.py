import cv2
import json
import os
import math, shutil
import numpy as np
import time
from tkinter.messagebox import showerror, showwarning, showinfo
from lib_main import edit_csv_tab, remove

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
eraser_shape = 'square'
show_help = True

text = {
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

def load_polygons_from_txt(img, file_path):
    global name_label
    all_polygons_data = []  # Thay đổi để lưu trữ danh sách các dictionary
    if len(img.shape) == 3:
        img_height, img_width, _ = img.shape
    else:
        img_height, img_width = img.shape
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
            center = calculate_center(poly)
            name_label[str(label_loaded)][1].append(center)
            all_polygons_data.append(polygon_entry)

    return all_polygons_data
# Function to save the current state
def save_current_state(file_path, current_image):
    with open(file_path, 'w') as f:
        f.write(current_image)
# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
# Function to load the current state
def load_current_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return None
# Mouse callback function
def draw_polygon(event, x, y, flags, param):
    global drawing, current_polygon, start_x, start_y, edit, name_label
    global label, mouse_pos, is_adding, eraser_size, eraser_shape, is_erasing, img_mask, polygons_data_list # Thêm polygons_data_list

    mouse_pos = [x, y]

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

while True:
    name_img = list_name_img[stt]
    # if name_img_old != name_img or ((stt == 0 or stt == len(list_name_img) - 1) and len(list_name_img) != 0):
    if name_img_old != name_img and len(list_name_img) != 0:
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
        polygons_data_list = load_polygons_from_txt(img_new, output_file) # Tải dữ liệu vào biến mới

        # Nếu có dữ liệu cũ, vẽ lại mask từ đó
        if polygons_data_list:
            for entry in polygons_data_list:
                poly = np.array(entry["polygon"], dtype=np.int32)
                if len(poly) > 0: # Ensure polygon is not empty
                    cv2.fillPoly(img_mask, [poly], 255) # Vẽ lại các polygon lên mask

        name_img_old = name_img

    if img_new is not None and img_mask is not None:
        img_new = np.ones((h2,w2+100,3), dtype=np.uint8) * 255
        img_new[:h2 ,:w2 ,:] = img_resize.copy()
    else:
        img_mask = np.zeros((500,500),dtype=np.uint8)
        print("zeros")
        img_resize = np.zeros((500,500,3),dtype=np.uint8)
        img_new = np.zeros((500,500+100,3),dtype=np.uint8) # Ensure it has enough width

    # Tìm các đường viền trên mask hiện tại
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Temporary list to store newly processed polygons with their labels
    updated_polygons_data_list = []

    # Map existing polygons to their labels
    # Create a dict for quick lookup if a polygon (or its approximate representation) already has a label
    existing_polygons_map = {}
    for entry in polygons_data_list:
        # Using a tuple of tuples for the polygon to make it hashable for dict key
        existing_polygons_map[tuple(map(tuple, entry["polygon"]))] = entry["label"]

    for contour_idx, contour in enumerate(contours):
        current_poly_tuple = tuple(map(tuple, [tuple(point[0]) for point in contour])) # Convert to tuple of tuples for hashing
        current_poly_list = [tuple(point[0]) for point in contour] # Keep as list of tuples for drawing/saving

        assigned_label = "none" # Default label

        # Check if this contour (or a very similar one) was previously labeled
        if current_poly_tuple in existing_polygons_map:
            assigned_label = existing_polygons_map[current_poly_tuple]
        else:
            # If not previously labeled, try to find a sample point inside it
            for label_key in name_label.keys():
                if label_key != "none":
                    points_for_current_label = name_label[label_key][1]
                    for pt_idx, pt in enumerate(points_for_current_label):
                        if cv2.pointPolygonTest(contour, (pt[0], pt[1]), False) >= 0:
                            assigned_label = label_key
                            break # Found a label for this contour
                if assigned_label != "none":
                    break # Stop checking other labels for this contour

        updated_polygons_data_list.append({"label": assigned_label, "polygon": current_poly_list})

    polygons_data_list = updated_polygons_data_list # Update the global list

    # Ensure all sample points in name_label are still within some contour
    # This part is tricky. A simple approach is to clear and repopulate based on current contours.
    # However, the user might manually place sample points for future manual drawing.
    # For now, let's assume sample points are mainly for initial contour assignment,
    # and direct labeling by clicking on a contour handles re-labeling.
    # If the intent is for sample points to "stick" to polygons, a more robust point-to-polygon tracking is needed.
    # For this request, we prioritize checking 'none' labels in polygons_data_list.

    img_new = img_new[:h2,:w2+100]

    # Tạo một lớp phủ tạm thời
    overlay = img_new.copy()

    # Màu sắc cho các nhãn, bạn có thể tùy chỉnh
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0)]
    alpha = 0.4  # Độ trong suốt (40%)

    for entry in polygons_data_list:
        poly_to_draw = np.array(entry["polygon"], dtype=np.int32)
        
        if entry["label"] != "none" and len(poly_to_draw) > 0:
            label_index = int(entry["label"])
            color = colors[label_index]
            
            # Vẽ hình đa giác đã tô màu lên lớp phủ
            cv2.fillPoly(overlay, [poly_to_draw], color)
            
            # # Vẽ đường viền để dễ nhìn hơn
            # cv2.polylines(img_new, [poly_to_draw], True, color, 2)
            
            # Tính toán và vẽ điểm trung tâm
            center = calculate_center(entry["polygon"])
            if center:
                cv2.circle(img_new, center, 5, (255, 255, 255), -1)
                # Hiển thị nhãn của polygon
                label_of_poly = name_label[entry["label"]][0]
                cv2.putText(img_new, label_of_poly, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # Hòa trộn lớp phủ với hình ảnh chính
    cv2.addWeighted(overlay, alpha, img_new, 1 - alpha, 0, img_new)


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
        # cv2.putText(img_new, display_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        showerror(title="error", message=display_message)
    else:
        display_message = "" # Clear message after duration

    if show_help:
        help_image = create_help_image(text)
        cv2.imshow('Help', help_image)
        if 'w2' in locals():
             cv2.moveWindow('Help', w2 + 120, 1)
        else:
             cv2.moveWindow('Help', 800, 1)

    cv2.imshow(name_img, img_new)
    cv2.imshow('mask ' + name_img, img_mask)
    cv2.moveWindow(name_img, 1, 1)
    cv2.setMouseCallback(name_img, draw_polygon)

    k = cv2.waitKey(1) & 0xFF

    # Check for unlabeled polygons before saving or navigating
    unlabeled_polygons_exist = False
    for entry in polygons_data_list:
        if entry["label"] == "none":
            unlabeled_polygons_exist = True
            break

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
        output_file = os.path.join(forder_output, name_img.split(".")[0] + ".txt")
        save_polygons_to_txt(img_mask, output_file, polygons_data_list)
        print(f"Polygons saved to {output_file}")
        polygons_data_list = [] # Xóa dữ liệu cho ảnh tiếp theo
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
        polygons_data_list = []
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