import cv2

img = cv2.imread(r"camera_data\anh_60.jpg")
if img is None:
    print("Không thể mở ảnh!")
    exit()

img_display = img.copy()

def click_event(event, x, y, flags, param):
    global img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        print(f"[{x}, {y}],")

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_event)

while True:
    cv2.imshow("Image", img_display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # nhấn ESC để thoát
        break

cv2.destroyAllWindows()
