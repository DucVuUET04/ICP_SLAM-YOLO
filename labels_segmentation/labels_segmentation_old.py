import cv2
import json
import os
import math, shutil
import numpy as np
 
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
polygons = []
polygons_copy = []
start_x, start_y = 0, 0
center = (0, 0)
number_resize = 5
state_file = 'current_state.txt'
edit = 0
 
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
def save_polygons_to_txt(img, file_path, polygons):
    if len(img.shape) == 3:
        img_height, img_width, _ = img.shape
    else:
        img_height, img_width = img.shape
    with open(file_path, 'w') as f:
        data = "0"
        for polygon in polygons:
            x = polygon[0] / img_width
            y = polygon[1] / img_height
            if x<0 : 
                x=0
            if x>1:
                x=1
            if y<0 :
                y=0
            if y>1:
                y=1
            data = data + " " + str(x) + " " + str(y)
        f.write(f"{data}")
 
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
if data_setting["forder_output"] == "None":
    forder_output = 'labels_segmentation/output'
else:
    forder_output = edit_path(data_setting["forder_output"])
number_resize = float(data_setting["resize"])
anpha = int(float(data_setting["anpha"]))
# Load image
folder_path = folder_input
path_output = forder_output
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
while True:
    name_img = list_name_img[stt]
    if name_img_old != name_img:
        output_file = os.path.join(path_output, name_img.split(".")[0] + ".txt")
        img = cv2.imread(os.path.join(folder_path, name_img))
        h, w, _ = img.shape
        img_copy = img.copy()
        img_copy = cv2.resize(img_copy, (int(w * number_resize), int(h * number_resize)))
        if os.path.exists(output_file):
            polygons = load_polygons_from_txt(img_copy, output_file)
        name_img_old = name_img
 
    img_copy = img.copy()
    img_copy = cv2.resize(img_copy, ((int(w * number_resize)), int(h * number_resize)))
    for pol in range(len(polygons)):
        cv2.circle(img_copy, polygons[pol], 3, (0, 0, 255), -1)
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
        output_file = os.path.join(path_output, name_img.split(".")[0] + ".txt")
        if len(polygons) > 0:
            save_polygons_to_txt(img_copy, output_file, polygons)
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
        if stt < len(list_name_img)-1:
            stt = stt + 1
        # Save polygons to txt format
        output_file = os.path.join(path_output, name_img.split(".")[0] + ".txt")
        if len(polygons) > 0:
            save_polygons_to_txt(img_copy, output_file, polygons)
            print(f"Polygons saved to {output_file}")
            polygons = []
    elif k == ord('a'):  # Press 'b' to rotate points
        if stt > 0:
            stt = stt - 1
        # Save polygons to txt format
        output_file = os.path.join(path_output, name_img.split(".")[0] + ".txt")
        if len(polygons) > 0:
            save_polygons_to_txt(img_copy, output_file, polygons)
            print(f"Polygons saved to {output_file}")
            polygons = []
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
        output_file = os.path.join(path_output, name_img.split(".")[0] + ".txt")
        if os.path.exists(output_file) == True:
            os.remove(output_file)
 
cv2.destroyAllWindows()
 
 