import cv2
import numpy as np
import os

def extract_color_histogram(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Histogram cho từng kênh BGR
    features = []
    for i in range(3):  # B, G, R
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    return np.array(features)

# Duyệt qua toàn bộ dataset
all_features = []
all_labels = []

for split in ["train", "val", "test"]:
    for class_name in os.listdir(f"data/{split}"):
        for img_file in os.listdir(f"data/{split}/{class_name}"):
            path = f"data/{split}/{class_name}/{img_file}"
            feat = extract_color_histogram(path)
            all_features.append(feat)
            all_labels.append(class_name)  # mệnh giá tiền (VD: "10000", "50000")

# Lưu vào file .npy
os.makedirs("data/features", exist_ok=True)
np.save("data/features/color.npy", np.array(all_features))
np.save("data/features/labels.npy", np.array(all_labels))

print("Đã lưu color features!")