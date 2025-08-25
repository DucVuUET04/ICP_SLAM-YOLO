from ultralytics import YOLO
import cv2
import numpy as np
# Load model OBB đã train
model = YOLO(r"C:\Xu_ly_anh\train7\weights\best.pt")
img_path = r"C:\Xu_ly_anh\camera_data_2\anh_278.jpg"
img = cv2.imread(img_path)

reference_box_coords = np.array([
    [152, 239],
    [480, 229],
    [478, 178],
    [150, 188]
], dtype=np.int32)

reference_center = tuple(np.mean(reference_box_coords, axis=0).astype(int))

cv2.polylines(img, [reference_box_coords], isClosed=True, color=(255, 0, 0), thickness=3)  # Màu xanh dương, dày hơn
cv2.putText(img, "Khung Goc", (reference_center[0] - 50, reference_center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.circle(img, reference_center, 5, (255, 0, 0), -1)  # Đánh dấu tâm khung gốc

results = model.predict(r"C:\Xu_ly_anh\camera_data_2\anh_220.jpg", task="obb", save=True, conf=0.5)

names = model.names


r= results[0]

boxes = r.obb.xyxyxyxy  # dạng (N, 4, 2) -> N hộp, mỗi hộp 4 điểm (x, y)
labels = r.obb.cls      # chỉ số class
confs = r.obb.conf      # độ tin cậy

for i in range(len(boxes)):
    
    coords = boxes[i].cpu().numpy().astype(np.int32)
    
 
    label = names[int(labels[i])]
    confidence = float(confs[i])
    area = cv2.contourArea(coords)

 
    detected_center = tuple(np.mean(coords, axis=0).astype(int))

   
    offset_x = detected_center[0] - reference_center[0]
    offset_y = detected_center[1] - reference_center[1]

 
    print(f"Label: {label}, Confidence: {confidence:.2f}, Area: {area:.0f} px^2")
    print(f"  Tam vat the: {detected_center}")
    print(f"  Do lech so voi goc: (dx={offset_x}, dy={offset_y})")
    print("-" * 30)

 
    cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.circle(img, detected_center, 5, (0, 255, 0), -1)

    cv2.line(img, reference_center, detected_center, (255, 255, 0), 1)
    text_lines = [
        # f"{label} {confidence:.2f}",
        # f"Area: {int(area)}",
        # f"Offset: ({offset_x}, {offset_y})"
    ]
    
    text_origin = (coords[0, 0], coords[0, 1] - 15)

    for j, line in enumerate(text_lines):
        y = text_origin[1] - j * 20

        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (text_origin[0], y - h), (text_origin[0] + w, y), (0, 0, 0), -1)
        cv2.putText(img, line, (text_origin[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

cv2.imshow("Kết quả trainning", img)
cv2.waitKey(0)
cv2.destroyAllWindows()