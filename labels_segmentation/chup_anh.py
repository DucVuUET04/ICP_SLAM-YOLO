import cv2
import os

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)      
cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

save_dir = r"camera_data"

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_counter = 0 # Biến đếm số ảnh
while True:
    ret, frame = cap.read()
    ret2,fram2 = cap2.read()
    if not ret:
        print("Không đọc được frame!")
        break

    cv2.imshow("USB Camera", frame)
    cv2.imshow("USB Camera2", fram2)
    key = cv2.waitKey(1) & 0xFF
   
    # Nhấn 'c' để chụp ảnh
    if key == ord('c'):
        img_name = os.path.join(save_dir, f"anh_1_{img_counter}.jpg")
        img_name_2 = os.path.join(save_dir, f"anh_2_{img_counter}.jpg")
        cv2.imwrite(img_name, frame)
        cv2.imwrite(img_name_2, fram2)
        print(f"Lưu ảnh {img_name}")
        img_counter += 1

    # Nhấn 'q' để thoát
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
