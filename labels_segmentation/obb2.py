import numpy as np
import math
import cv2

from ultralytics import YOLO
cap= cv2.VideoCapture(0)
def analyze_object_pose(coords, image_shape):

    img_height, img_width = image_shape[:2]

    coords_sorted_y = sorted(coords, key=lambda p: p[1])
    top_points = sorted(coords_sorted_y[:2], key=lambda p: p[0])
    bottom_points = sorted(coords_sorted_y[2:], key=lambda p: p[0])

    if len(top_points) < 2 or len(bottom_points) < 2:
        return "Không xác định", "Không xác định", 0

    tl, tr = top_points[0], top_points[1]
    bl, br = bottom_points[0], bottom_points[1]

    center_point = np.mean(coords, axis=0)
    center_x_threshold = img_width * 0.15

    if center_point[0] < (img_width / 2 - center_x_threshold):
        position_status = "Lệch Trái"
    elif center_point[0] > (img_width / 2 + center_x_threshold):
        position_status = "Lệch Phải"
    else:
        position_status = "Chính Giữa"

    left_side_len = np.linalg.norm(tl - bl)
    right_side_len = np.linalg.norm(tr - br)

    if right_side_len < 1e-6:
        rotation_status = "Không xác định"
    else:
        side_len_ratio = left_side_len / right_side_len
        if side_len_ratio > 1.2:
            rotation_status = "Xoay Phải"
        elif side_len_ratio < 0.8:
            rotation_status = "Xoay Trái"
        else:
            rotation_status = "Vuông Góc"

    bottom_vector = br - bl
    roll_angle = math.degrees(math.atan2(bottom_vector[1], bottom_vector[0]))

    return position_status, rotation_status, roll_angle
def tim_vitri(coords):
  
    coords_sorted_y = sorted(coords, key=lambda p: p[1])

    top_points = sorted(coords_sorted_y[:2], key=lambda p: p[0])
    # Sắp xếp các điểm dưới theo tọa độ x (phải sang trái để có thứ tự tl, tr, br, bl)
    bottom_points = sorted(coords_sorted_y[2:], key=lambda p: p[0], reverse=True)

    if len(top_points) < 2 or len(bottom_points) < 2:
        return "Khong xac dinh", 0

    tl, tr = top_points[0], top_points[1]
    br, bl = bottom_points[0], bottom_points[1] 

    print (f"tl = {tl}, tr = {tr}, br = {br}, bl = {bl}")

    l_mid_x = (tl[0]+ bl[0])//2
    l_mid_y = (tl[1]+ bl[1])//2
    r_mid_x = (tr[0]+ br[0])//2
    r_mid_y = (tr[1]+ br[1])//2
    l_mid = (l_mid_x, l_mid_y)
    r_mid = (r_mid_x, r_mid_y)
    print(f"l_mid = {l_mid}, r_mid = {r_mid}")
    scale_x = (640-r_mid[0])/(l_mid[0])
    scale_y = int(tl[1])/int(tr[[1]])
    print(f"scale_x = {scale_x}")
    print(f"scale_y = {scale_y}")
    
    # canh_tren = np.linalg.norm(tl-tr)
    canh_duoi = np.linalg.norm(bl-br)
    # delta_y= (scale_y-1)* tl[1]
    delta_y_2 = (bl[1]-br[1])
    angle= math.degrees(math.asin(delta_y_2/canh_duoi))
    
    if  scale_x<=0.8:
        if scale_y>=1.12:
            position_status = "Chinh Giua"
            rotation_status= "xoay trai"
        else:
            position_status = "Lech Phai"
            rotation_status= "Vuong Goc"
  
    elif scale_x>= 1.2:
        if scale_y>=1.12:
            position_status = "Chinh Giua"
            rotation_status= "xoay phai"
        else:
            position_status = "Lech Trai"
            rotation_status= "Vuong Goc"
    else:
        position_status = "Chinh Giua"
        rotation_status= "Vuong Goc"

 

    return position_status, rotation_status, angle
def estimate_3d_pose(image_points, object_real_dims, camera_matrix, dist_coeffs):
    obj_w, obj_h = object_real_dims

    object_points_3d = np.array([
        [0, obj_h, 0],
        [obj_w, obj_h, 0],
        [obj_w, 0, 0],
        [0, 0, 0],
    ], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(object_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)

    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    proj_matrix = np.hstack((rotation_matrix, tvec))
    
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    return rvec, tvec, euler_angles.flatten()

def draw_axes(img, rvec, tvec, camera_matrix, dist_coeffs, length=50):
    axis_points_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
    axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    axis_points_2d = np.int32(axis_points_2d).reshape(-1, 2)
    
    cv2.line(img, tuple(axis_points_2d[0]), tuple(axis_points_2d[1]), (255, 0, 0), 3)
    cv2.line(img, tuple(axis_points_2d[0]), tuple(axis_points_2d[2]), (0, 255, 255), 3)
    cv2.line(img, tuple(axis_points_2d[0]), tuple(axis_points_2d[3]), (0, 0, 255), 3)

def main():
    model = YOLO(r"C:\Xu_ly_anh\train7\weights\best.pt")
    # img_path = r"C:\Xu_ly_anh\camera_data_2\anh_123.jpg"
    ret, frame = cap.read()
    
    cv2.imshow("USB Camera", frame)

    img = cv2.imread(frame)
    if img is None:
     
        return

    OBJECT_REAL_DIMS = (110, 15)

    h, w = img.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

    results = model.predict(img, task="obb", save=False, conf=0.5, verbose=False)

    names = model.names
    if not results:
        print("Không có kết quả dự đoán.")
        return
   
    r = results[0]
    boxes = r.obb.xyxyxyxy
    labels = r.obb.cls
    confs = r.obb.conf

    for i in range(len(boxes)):
        coords = boxes[i].cpu().numpy().astype(np.int32)
        label = names[int(labels[i])]
        position_status, rotation_status, angle = tim_vitri(coords)

        print(f"  --> Trang thai vi tri: {position_status}")
        print(f"  --> Trang thai xoay: {rotation_status}")
        # print(f"  --> Goc xoay: {angle:.2f} do")
        print("-" * 30)
        coords_sorted_y = sorted(coords, key=lambda p: p[1])
        top_points = sorted(coords_sorted_y[:2], key=lambda p: p[0])
        bottom_points = sorted(coords_sorted_y[2:], key=lambda p: p[0])
        tl, tr = top_points[0], top_points[1]
        bl, br = bottom_points[0], bottom_points[1]
        
        image_points_pnp = np.array([tl, tr, br, bl], dtype=np.float32)

        rvec, tvec, euler_angles = estimate_3d_pose(image_points_pnp, OBJECT_REAL_DIMS, camera_matrix, dist_coeffs)

        print(f"Label: {label}, Confidence: {float(confs[i]):.2f}")
        if euler_angles is not None:
            pitch, yaw, roll = euler_angles
            print(f"  --> Góc xoay 3D: Pitch={pitch:.2f}, Yaw={yaw:.2f}, Roll={roll:.2f} độ")
            print(f"  --> Vị trí 3D: X={(tvec[0][0]+55)}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f} mm")
            draw_axes(img, rvec, tvec, camera_matrix, dist_coeffs)
        else:
            print("  -->")

        print("-" * 30)

        cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 0), thickness=2)

        center_point = tuple(np.mean(coords, axis=0).astype(int))

        text_lines = [
            # f"Pitch: {euler_angles[0]:.1f}",
            f"X={tvec[0][0]+55}",
            f"Yaw: {euler_angles[1]:.1f}",
            # f"Roll: {euler_angles[2]:.1f}"
        ]
        
        y_pos = center_point[1] - 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        for line in reversed(text_lines):
            (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            box_coords = ((center_point[0] - w // 2 - 5, y_pos - h - 5), (center_point[0] + w // 2 + 5, y_pos + 5))
            cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), -1)
            cv2.putText(img, line, (center_point[0] - w // 2, y_pos), font, font_scale, (0, 0, 255), font_thickness)
            y_pos -= (h + 10)

    cv2.imshow("Ket qua phan tich", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO(r"C:\Xu_ly_anh\train7\weights\best.pt")
    # img_path = r"C:\Xu_ly_anh\camera_data_2\anh_123.jpg"
    while True:
        ret, frame = cap.read()
        if ret ==True:
            # cv2.imshow("USB Camera", frame)

            img = frame
          
            OBJECT_REAL_DIMS = (110, 15)

            h, w = img.shape[:2]
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))

            results = model.predict(img, task="obb", save=False, conf=0.5, verbose=False)

            names = model.names
           
        
            r = results[0]
            boxes = r.obb.xyxyxyxy
            labels = r.obb.cls
            confs = r.obb.conf

            for i in range(len(boxes)):
                coords = boxes[i].cpu().numpy().astype(np.int32)
                label = names[int(labels[i])]
                position_status, rotation_status, angle = tim_vitri(coords)

                print(f"  --> Trang thai vi tri: {position_status}")
                print(f"  --> Trang thai xoay: {rotation_status}")
                # print(f"  --> Goc xoay: {angle:.2f} do")
                print("-" * 30)
                coords_sorted_y = sorted(coords, key=lambda p: p[1])
                top_points = sorted(coords_sorted_y[:2], key=lambda p: p[0])
                bottom_points = sorted(coords_sorted_y[2:], key=lambda p: p[0])
                tl, tr = top_points[0], top_points[1]
                bl, br = bottom_points[0], bottom_points[1]
                
                image_points_pnp = np.array([tl, tr, br, bl], dtype=np.float32)

                rvec, tvec, euler_angles = estimate_3d_pose(image_points_pnp, OBJECT_REAL_DIMS, camera_matrix, dist_coeffs)

                print(f"Label: {label}, Confidence: {float(confs[i]):.2f}")
                if euler_angles is not None:
                    pitch, yaw, roll = euler_angles
                    print(f"  --> Góc xoay 3D: Pitch={pitch:.2f}, Yaw={yaw:.2f}, Roll={roll:.2f} độ")
                    print(f"  --> Vị trí 3D: X={(tvec[0][0]+55)}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f} mm")
                    draw_axes(img, rvec, tvec, camera_matrix, dist_coeffs)
                else:
                    print("  -->")

                print("-" * 30)

                cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 0), thickness=2)

                center_point = tuple(np.mean(coords, axis=0).astype(int))

                text_lines = [
                    
                    f"X={tvec[0][0]+55}",
                    f"Z= {tvec[2][0]}",
                    f"Yaw: {euler_angles[1]:.1f}",
                    # f"Pitch: {euler_angles[0]:.1f}",
                    # f"Roll: {euler_angles[2]:.1f}"
                ]
                
                y_pos = center_point[1] - 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                for line in reversed(text_lines):
                    (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                    box_coords = ((center_point[0] - w // 2 - 5, y_pos - h - 5), (center_point[0] + w // 2 + 5, y_pos + 5))
                    cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), -1)
                    cv2.putText(img, line, (center_point[0] - w // 2, y_pos), font, font_scale, (0, 0, 255), font_thickness)
                    y_pos -= (h + 10)

            cv2.imshow("Ket qua phan tich", img)
            key = cv2.waitKey(1) & 0xFF
        
            if key == ord('q') or cv2.getWindowProperty('Ket qua phan tich', cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()

        