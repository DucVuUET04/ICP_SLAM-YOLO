import numpy as np
from ultralytics import YOLO
import cv2
import math
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)      
cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  


def stereo_to_3d(points_left, points_right, f, cx, cy, B):
    points_left = np.array(points_left, dtype=float)
    points_right = np.array(points_right, dtype=float)

    disparity = np.abs(points_left[:,0] - points_right[:,0])
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


    mean_depth = np.mean(corners_3d[:,2])

    return normal, angle_yaw_rad, mean_depth
if __name__ == '__main__':
    model = YOLO(r"C:\Xu_ly_anh\train2\weights\best.pt")
    img_gray = np.full((420, 640), 150, dtype=np.uint8)

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
                corners_3d = stereo_to_3d(points_left_corners, points_right_corners, f, cx, cy, B)
                normal, angle_rad, mean_depth = pallet_orientation_and_distance(corners_3d)
                angle_deg = -np.degrees(angle_rad)
                pallet_center_3d = np.mean(corners_3d, axis=0)
                angle_horizontal_rad = np.arctan2(pallet_center_3d[0], pallet_center_3d[2])
                angle_horizontal_deg = np.degrees(angle_horizontal_rad)
                # distance_x =13-mean_depth * math.tan(angle_horizontal_rad)
                x_pallet_mm= 110 
               
                palet_img_x=x_pallet_mm* math.cos(angle_rad)
                px_mm= delta_x_pixel/palet_img_x

                dis_lech= (delta_x / px_mm)-13
                print("---")
                # print("Pháp tuyến pallet:", normal)
                print(f"Góc nghiêng Pallet: {angle_deg:.2f}°")  
                print(f"Góc lệch ngang (so với tâm camera): {(angle_horizontal_deg):.2f}°")
                print(f"Khoảng cách tới Pallet: {mean_depth:.2f} mm")
                print(f"Độ lệch camera: {dis_lech:.2f} mm") 
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
            except Exception as e:
                print(f"Lỗi tính toán 3D: {e}")

        cv2.imshow("Image left", img_1)
        cv2.imshow("Image right", img_2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
