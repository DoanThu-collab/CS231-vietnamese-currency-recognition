import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

src_dir = r"D:\UIT_NamHai\NHAP_MON_CV\CS231-vietnamese-currency-recognition\data\raw"
output_dir = "data"

IMG_SIZE = (224, 224)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]

    if len(images) == 0:
        continue

    train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    def process_and_save(file_list, split):
        for f in file_list:
            src = os.path.join(class_path, f)
            dst = os.path.join(output_dir, split, class_name, f)

            try:
                img = Image.open(src).convert("RGB")  # đảm bảo 3 channels
                img = img.resize(IMG_SIZE)

                img.save(dst)

            except Exception as e:
                print(f"Lỗi file: {src} | {e}")

    process_and_save(train_files, "train")
    process_and_save(val_files, "val")
    process_and_save(test_files, "test")

print("Done resizing + splitting!")