import os
import glob

def check_and_fix_labels(labels_dir, fix=False):
    """
    Checks for out-of-bounds coordinates in YOLO label files and optionally fixes them.

    Args:
        labels_dir (str): Path to the directory containing the label .txt files.
        fix (bool): If True, clamps out-of-bounds coordinates to the [0, 1] range.
    """
    if not os.path.isdir(labels_dir):
        print(f"Error: Directory not found at '{labels_dir}'")
        return

    print(f"Scanning labels in: {labels_dir}")
    found_issues = False
    
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))

    for label_file in label_files:
        is_corrupt = False
        fixed_lines = []
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = parts[0]
                coords = [float(p) for p in parts[1:]]
                
                # Check for out-of-bounds coordinates
                if any(c < 0.0 or c > 1.0 for c in coords):
                    is_corrupt = True
                    found_issues = True
                    print(f"WARNING: Found out-of-bounds coordinate in {os.path.basename(label_file)} on line {i+1}:")
                    print(f"  Original line: {line.strip()}")
                    
                    if fix:
                        # Clamp coordinates to [0.0, 1.0]
                        clamped_coords = [max(0.0, min(1.0, c)) for c in coords]
                        fixed_line = f"{class_id} {' '.join(map(str, clamped_coords))}\n"
                        fixed_lines.append(fixed_line)
                        print(f"  Fixed line:    {fixed_line.strip()}")
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)

            if is_corrupt and fix:
                with open(label_file, 'w') as f:
                    f.writelines(fixed_lines)
                print(f"  -> Fixed and saved {os.path.basename(label_file)}\n")

        except Exception as e:
            print(f"Error processing file {label_file}: {e}")

    if not found_issues:
        print("Scan complete. No issues found.")

if __name__ == '__main__':
    # Paths inferred from your training logs.
    validation_labels_path = r'D:\tupn\data_training\duc_pallet\dataset_yolo\val\labels'
    training_labels_path = r'D:\tupn\data_training\duc_pallet\dataset_yolo\train\labels'

    print("--- Checking and fixing validation labels ---")
    check_and_fix_labels(validation_labels_path, fix=True)

    print("\n--- Checking and fixing training labels ---")
    # It's a good practice to check the training set as well.
    check_and_fix_labels(training_labels_path, fix=True)

    print("\nLabel check and fix process complete.")
