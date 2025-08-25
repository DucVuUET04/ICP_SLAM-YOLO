import cv2
import json
import os
import math, shutil
import numpy as np
from lib_main import load_data_csv, edit_csv_tab, remove
 
 
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
file_path_2 = "kiem_tra.csv"
edit_csv_tab.new_csv_replace(file_path_2,["name","x00","y00","x10","y10","x11","y11","x01","y01"])
# Initialize global variables
drawing = False
current_polygon = []
polygons = []
polygons_copy = []
start_x, start_y = 0, 0
center = (0, 0)
number_resize = 5
state_file = 'current_state.txt'
edit = 0
 
pose_v_number = -1
list_pose_v_number = [2, 2, 2, 2]
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
def save_polygons_to_txt(img, file_path_obb, file_path_pose, file_path_oject, polygons):
   
    if len(img.shape) == 3:
        img_height, img_width, _ = img.shape
    else:
        img_height, img_width = img.shape
    with open(file_path_obb, 'w') as f:
        data = "0"
        for polygon in polygons:
            print(polygon)
            x = polygon[0] / img_width
            y = polygon[1] / img_height
            if x > 1:
                x = 1
            if y > 1:
                y = 1
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            data = data + " " + str(x) + " " + str(y)
        f.write(f"{data}")
    # Save pose data
    x1,y1,x2,y2 = [0,0,0,0]
    toa_do_x = []
    toa_do_y = []
    for polygon in polygons:
        x = polygon[0] / img_width
        y = polygon[1] / img_height
        if x > 1:
            x = 1
        if y > 1:
            y = 1
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        toa_do_x.append(x)
        toa_do_y.append(y)
    # tinh khung bao quanh
    if len(toa_do_x) > 0:
        x1 = min(toa_do_x)
        x2 = max(toa_do_x)
        y1 = min(toa_do_y)
        y2 = max(toa_do_y)
        # tâm của khung bao
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # width và height của khung bao
        width = x2 - x1
        height = y2 - y1
        # pose data
        with open(file_path_pose, 'w') as f:
            data2 = "0" + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height)
            for i in range(len(toa_do_x)):
                data2 += " " + str(toa_do_x[i]) + " " + str(toa_do_y[i]) + " " + str(list_pose_v_number[i])
                print("list_pose_v_number[i]", list_pose_v_number[i])
            f.write(f"{data2}")
 
    # Save object data
    if len(toa_do_x) > 0:
        x1 = min(toa_do_x)
        x2 = max(toa_do_x)
        y1 = min(toa_do_y)
        y2 = max(toa_do_y)
        # tâm của khung bao
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # width và height của khung bao
        width = x2 - x1
        height = y2 - y1
        # pose data
        with open(file_path_oject, 'w') as f:
            data_oject = "0" + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height)
            f.write(f"{data_oject}")
     
    data2 = [file_path_obb]
    print("save")
    for polygon in polygons:
        print("polygon", polygon)
        x2 = polygon[0] / number_resize
        y2 = polygon[1] / number_resize
        data2.append(str(int(x2)))
        data2.append(str(int(y2)))
    edit_csv_tab.append_csv(file_path_2, data2)
 
 
# Function to load polygons from a TXT file
def load_polygons_from_txt(img, file_path):
    polygons = []
    if len(img.shape) == 3:
        img_height, img_width, _ = img.shape
    else:
        img_height, img_width = img.shape
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()[1:]  # Skip the first element (label)
            for i in range(0, len(parts), 2):
                x = float(parts[i]) * img_width
                y = float(parts[i + 1]) * img_height
                polygons.append((int(x), int(y)))
    return polygons
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
    global drawing, current_polygon, polygons, start_x, start_y
 
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
 
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
 
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if edit == 0:
            for i in range(len(current_polygon)):
                polygons.append(current_polygon[i])
            current_polygon = []
        else:
            distance0 = 1000
            index = 0
            for pol in range(0,len(polygons)):
                diem1 = polygons[pol]
                diem2 = [x,y]
                distance = calculate_distance(diem1, diem2)
                if distance < distance0:
                    index = pol
                    distance0 = distance
            if distance0 != 1000:
                polygons[index] = [x,y]
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            dx = x - start_x
            dy = y - start_y
            polygon = []
            for pol in range(len(polygons)):
                px, py = polygons[pol]
                px += dx
                py += dy
                polygon.append([px, py])
            polygons = polygon
            start_x, start_y = x, y
 
 
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
 
folder_input = edit_path(data_setting["folder_input"])
if data_setting["forder_output_obb"] == "None":
    forder_output_obb = 'labels_segmentation/output_obb'
else:
    forder_output_obb = edit_path(data_setting["forder_output_obb"])
 
if data_setting["forder_output_pose"] == "None":
    forder_output_pose = 'labels_segmentation/output_pose'
else:
    forder_output_pose = edit_path(data_setting["forder_output_pose"])
 
if data_setting["forder_output_oject"] == "None":
    forder_output_oject = 'labels_segmentation/output_oject'
else:
    forder_output_oject = edit_path(data_setting["forder_output_oject"])
 
 
number_resize = float(data_setting["resize"])
anpha = int(float(data_setting["anpha"]))
# Load image
folder_path = remove.tao_folder(folder_input)
path_output_obb = remove.tao_folder(forder_output_obb)
path_output_pose = remove.tao_folder(forder_output_pose)
path_output_oject = remove.tao_folder(forder_output_oject)
 
list_name_img = os.listdir(folder_path)
print(list_name_img)
stt = 0
name_img_old = ""
 
# Load the last checked image state
last_checked_image = load_current_state(state_file)
if last_checked_image and last_checked_image in list_name_img:
    stt = list_name_img.index(last_checked_image)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_polygon)
name = 1
load_max = 0
while True:
   
    name_img = list_name_img[stt]
    # name_img = str(name) + ".jpg"
    if name_img_old != name_img or (((stt == 0 or stt == len(list_name_img) - 1) and len(list_name_img) != 0) and load_max == 0):
        print(name_img)
        load_max = 1
        output_file = os.path.join(path_output_obb, name_img.split(".")[0] + ".txt")
        img = cv2.imread(os.path.join(folder_path, name_img))
        h, w, _ = img.shape
        img_copy = img.copy()
        img_copy = cv2.resize(img_copy, (int(w * number_resize), int(h * number_resize)))
        if os.path.exists(output_file):
            polygons = load_polygons_from_txt(img_copy, output_file)
        name_img_old = name_img
 
    img_copy = img.copy()
    img_copy = cv2.resize(img_copy, ((int(w * number_resize)), int(h * number_resize)))
 
    name_circle = 0
    for i in range(len(list_pose_v_number)):
        cv2.putText(img_copy, str(list_pose_v_number[i]), (10 + i * 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    for pol in range(len(polygons)):
        cv2.circle(img_copy, polygons[pol], 3, (0, 0, 255), -1)
        cv2.putText(img_copy, str(name_circle), (polygons[pol][0] + 5, polygons[pol][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        name_circle += 1
        if len(polygons) > 1:
            if pol < len(polygons) - 1:
                cv2.line(img_copy, polygons[pol], polygons[pol + 1], (0, 255, 0), 2)
            else:
                cv2.line(img_copy, polygons[pol], polygons[0], (0, 255, 0), 2)
 
    # Calculate and draw center point
    center = calculate_center(polygons)
    if center:
        cv2.circle(img_copy, center, 5, (255, 0, 0), -1)
 
    cv2.imshow('image', img_copy)
    k = cv2.waitKey(5) & 0xFF
 
    if k == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:  # Press 'q' to exit
        save_current_state(state_file, name_img)
        # Save polygons to txt format
        output_file_obb = os.path.join(path_output_obb, name_img.split(".")[0] + ".txt")
        output_file_pose = os.path.join(path_output_pose, name_img.split(".")[0] + ".txt")
        output_file_oject = os.path.join(path_output_oject, name_img.split(".")[0] + ".txt")
        if len(polygons) > 0:
            save_polygons_to_txt(img_copy, output_file_obb, output_file_pose, output_file_oject, polygons)
        break
    elif k == ord('n'):  # Press 'b' to rotate points
        if center:
            angle = 5  # Rotate by 10 degrees
            polygons = [rotate_point(p, center, angle) for p in polygons]
    elif k == ord('b'):  # Press 'b' to rotate points
        if center:
            angle = -5  # Rotate by 10 degrees
            polygons = [rotate_point(p, center, angle) for p in polygons]
    elif k == ord('d'):  # Press 'b' to rotate points
       
        load_max = 0
        name = name + 1
        if stt < len(list_name_img)-1:
            stt = stt + 1
        # Save polygons to txt format
        output_file_obb = os.path.join(path_output_obb, name_img.split(".")[0] + ".txt")
        output_file_pose = os.path.join(path_output_pose, name_img.split(".")[0] + ".txt")
        output_file_oject = os.path.join(path_output_oject, name_img.split(".")[0] + ".txt")
        if len(polygons) > 0:
            save_polygons_to_txt(img_copy, output_file_obb, output_file_pose, output_file_oject, polygons)
            print(f"Polygons saved to {output_file}")
            polygons = []
        pose_v_number = -1
        list_pose_v_number = [2, 2, 2, 2]
    elif k == ord('a'):  # Press 'b' to rotate points
       
        load_max = 0
        if stt > 0:
            stt = stt - 1
        # Save polygons to txt format
        output_file_obb = os.path.join(path_output_obb, name_img.split(".")[0] + ".txt")
        output_file_pose = os.path.join(path_output_pose, name_img.split(".")[0] + ".txt")
        output_file_oject = os.path.join(path_output_oject, name_img.split(".")[0] + ".txt")
        if len(polygons) > 0:
            save_polygons_to_txt(img_copy, output_file_obb, output_file_pose, output_file_oject, polygons)
            print(f"Polygons saved to {output_file}")
            polygons = []
        pose_v_number = -1
        list_pose_v_number = [2, 2, 2, 2]
    elif k == ord('z'):  # Press 'b' to rotate points
        if len(polygons) > 0:
            del polygons[-1]
    elif k == ord('c'):
        polygons_copy = polygons
    elif k == ord('v'):
        polygons = polygons_copy
    elif k == ord('e'):
        if edit == 0:
            edit = 1
        else:
            edit = 0
    elif k == ord('r'):
        polygons = []
        # Save polygons to txt format
        output_file_obb = os.path.join(path_output_obb, name_img.split(".")[0] + ".txt")
        output_file_pose = os.path.join(path_output_pose, name_img.split(".")[0] + ".txt")
        output_file_oject = os.path.join(path_output_oject, name_img.split(".")[0] + ".txt")
        if os.path.exists(output_file_obb) == True:
            os.remove(output_file_obb)
        if os.path.exists(output_file_pose) == True:
            os.remove(output_file_pose)
        if os.path.exists(output_file_oject) == True:
            os.remove(output_file_oject)
    if pose_v_number == -1:
        if k == ord('0'):
            pose_v_number = 0
        elif k == ord('1'):
            pose_v_number = 1
        elif k == ord('2'):
            pose_v_number = 2
        elif k == ord('3'):
            pose_v_number = 3
    else:
        if k == ord('0'):
            list_pose_v_number[pose_v_number] = 0
        elif k == ord('1'):
            list_pose_v_number[pose_v_number] = 1
        elif k == ord('2'):
            list_pose_v_number[pose_v_number] = 2
cv2.destroyAllWindows()
 
 
 