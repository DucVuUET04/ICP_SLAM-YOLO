import os
import shutil
import random

def split_yolo_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    # Tạo thư mục đích
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()
    
    # Xáo trộn
    random.shuffle(image_files)
    
    # Chia train / val
    train_count = int(len(image_files) * train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    def copy_files(file_list, split):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            
            src_img_path = os.path.join(image_dir, img_file)
            src_label_path = os.path.join(label_dir, label_file)
            
            dst_img_path = os.path.join(output_dir, split, 'images', img_file)
            dst_label_path = os.path.join(output_dir, split, 'labels', label_file)
            
            shutil.copy2(src_img_path, dst_img_path)
            
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
            else:
                print(f"[Warning] Không tìm thấy label cho {img_file}")
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    print(f"✅ Hoàn thành! {len(train_files)} ảnh train, {len(val_files)} ảnh val.")

# Ví dụ:
split_yolo_dataset(
    image_dir="camera_data",
    label_dir="output_oject",
    output_dir="dataset_pallet",
    train_ratio=0.8
)
