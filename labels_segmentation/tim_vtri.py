import numpy as np
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  


def stereo_to_3d(points_left, points_right, f, cx, cy, B):
    points_left = np.array(points_left, dtype=float)
    points_right = np.array(points_right, dtype=float)

    disparity = points_left[:,0] - points_right[:,0]
    disparity[disparity == 0] = 1e-6

    Z = (f * B) / disparity
    X = ((points_left[:,0] - cx) * Z) / f
    Y = ((points_left[:,1] - cy) * Z) / f

    return np.vstack((X, Y, Z)).T

def pallet_orientation_and_distance(corners_3d):
    """
    Tính toán hướng và khoảng cách của pallet.

    Args:
        corners_3d: Mảng numpy chứa tọa độ 3D của các góc pallet.
        holes_3d: Mảng numpy chứa tọa độ 3D của các lỗ trên pallet.
    Returns:
        normal, angle_deg, mean_depth, dist_hole: Pháp tuyến, góc nghiêng, khoảng cách trung bình, khoảng cách tới lỗ.
    """
    v1 = corners_3d[1] - corners_3d[0]
    v2 = corners_3d[2] - corners_3d[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    if normal[2] < 0:
        normal = -normal

    angle_yaw_rad = np.arctan2(normal[0], normal[2])
    angle_deg = -np.degrees(angle_yaw_rad)

    mean_depth = np.mean(corners_3d[:,2])

    return normal, angle_deg, mean_depth
if __name__ == '__main__':
    model = YOLO(r"C:\Xu_ly_anh\train9\weights\best.pt")

    B = 26.0 
    f = 381
    cx, cy = 320, 240

    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()

        if not ret or not ret2:
            print("Không thể đọc frame từ camera. Thoát...")
            break

        img_1 = frame.copy()
        img_2 = frame2.copy()

        results_1 = model.predict(img_1, task="obb", save=False, conf=0.5, verbose=False)
        results_2 = model.predict(img_2, task="obb", save=False, conf=0.5, verbose=False)

        points_right_corners = None
        points_left_corners = None

        # r = results_1[0]
        # boxes = r.obb.xyxyxyxy
        # if len(boxes) > 0:
        #     confs = r.obb.conf
        #     labels = r.obb.cls
        #     names = model.names
        #     max_conf_index = np.argmax(confs)
        #     coords = boxes[max_conf_index].cpu().numpy().astype(np.int32)
            
        #     points_right_corners = coords.astype(np.float32)
        #     cv2.polylines(img_1, [coords], isClosed=True, color=(0, 255, 0), thickness=2)

        # r2 = results_2[0]
        # boxes2 = r2.obb.xyxyxyxy
        # if len(boxes2) > 0:
        #     confs2 = r2.obb.conf
        #     labels2 = r2.obb.cls
        #     names2 = model.names
        #     max_conf_index2 = np.argmax(confs2)
        #     coords2 = boxes2[max_conf_index2].cpu().numpy().astype(np.int32)
            
        #     points_left_corners = coords2.astype(np.float32)
        #     cv2.polylines(img_2, [coords2], isClosed=True, color=(0, 255, 0), thickness=2)
        r = results_1[0]
        boxes = r.boxes  

        if len(boxes) > 0:
            box = boxes[0]  
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            points_right_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            cv2.polylines(img_1, [points_right_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
        r2 = results_2[0]
        boxes2 = r2.boxes  

        if len(boxes2) > 0:
            box = boxes[0]  
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            points_left_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            cv2.polylines(img_1, [points_left_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
        

        if points_left_corners is not None and points_right_corners is not None:
            try:
                corners_3d = stereo_to_3d(points_left_corners, points_right_corners, f, cx, cy, B)
                normal, angle_deg, mean_depth = pallet_orientation_and_distance(corners_3d)
                pallet_center_3d = np.mean(corners_3d, axis=0)
                angle_horizontal_rad = np.arctan2(pallet_center_3d[0], pallet_center_3d[2])
                angle_horizontal_deg = np.degrees(angle_horizontal_rad)

                print("---")
                print("Pháp tuyến pallet:", normal)
                print(f"Góc nghiêng (so với trục Z): {angle_deg:.2f}°")  
                print(f"Khoảng cách (theo Z): {mean_depth:.2f} mm")
            except Exception as e:
                print(f"Lỗi tính toán 3D: {e}")

        cv2.imshow("Image left", img_1)
        cv2.imshow("Image right", img_2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
