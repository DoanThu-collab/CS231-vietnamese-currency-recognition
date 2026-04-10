import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# --- CẤU HÌNH ĐƯỜNG DẪN & THIẾT BỊ ---
# 1. Tên mô hình đã train xong
model_path = "cnn/cnn_model.h5" 
# 2. File mapping thông tin
mapping_path = "app/mapping.json" 
# 3. Danh sách 5 ảnh nhóm muốn test (nhớ đổi tên ảnh cho đúng)
image_list = [
    "VNĐ1.webp",  # Tương ứng image_6.png
    "VNĐ2.webp",          # Tương ứng image_7.png
    "VNĐ3.jpg",        # Tương ứng image_8.png
    "VNĐ4.webp",# Tương ứng image_9.png
    "VNĐ5.webp"        # Tương ứng image_10.png
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LUY Ý CỰC KỲ QUAN TRỌNG: DANH SÁCH LỚP ---
# PyTorch ImageFolder sắp xếp lớp theo bảng chữ cái ABC của tên thư mục.
# Dựa trên dataset của nhóm, thứ tự CHUẨN phải như sau. 
# Nhóm ĐỪNG SỬA thứ tự này trừ khi dataset thay đổi.
class_names = [
    '000200', '000500', '001000', '002000', '005000', 
    '010000', '020000', '050000', '100000', '200000', 
    '500000', 'noise'
]
num_classes = len(class_names)

# --- 1. TẠI LẠI CẤU TRÚC MÔ HÌNH (Phải khớp 100% lúc train) ---
# Tải cấu trúc ResNet18
model = models.resnet18() 

# Thay đổi FC layer khớp với bản ANTI-OVERFITTING (Sequential + Dropout)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, num_classes)
)

# --- 2. LOAD TRỌNG SỐ ĐÃ TRAIN & CHUYỂN SANG CHẾ ĐỘ EVAL ---
print(f"Đang tải mô hình từ: {model_path}...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
# CRITICAL: Chuyển mô hình sang chế độ đánh giá (tắt Dropout)
model.eval() 

# --- 3. ĐỊNH NGHĨA PREPROCESSING (Phải khớp tập Val/Test lúc train) ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. LOAD MAPPING JSON ---
with open(mapping_path, 'r', encoding='utf-8') as f:
    mapping = json.load(f)

def predict_and_show_info(image_path):
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file {image_path}")
        return

    # A. Load và tiền xử lý ảnh
    img = Image.open(image_path).convert('RGB') # Đảm bảo ảnh luôn có 3 kênh màu
    img_t = preprocess(img)
    # Thêm batch dimension (PyTorch model mong đợi batch 4D: BxCxHxW)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # B. Dự đoán
    with torch.no_grad(): # Tắt tính toán gradient để tăng tốc
        outputs = model(batch_t)
    
    # C. Lấy lớp có điểm cao nhất
    _, index = torch.max(outputs, 1)
    predicted_folder_name = class_names[index[0]]

    # D. Tra cứu thông tin từ mapping.json
    if predicted_folder_name == 'noise':
        denom = "Nhiễu/Không phải tiền VN"
        desc = "Mô hình xác định đây là ảnh nhiễu hoặc đối tượng khác không phải tiền Việt Nam được hỗ trợ."
    else:
        info = mapping.get(predicted_folder_name, {})
        denom = info.get('denomination', 'Không xác định')
        desc = info.get('description', 'Không có mô tả.')

    # E. Hiển thị kết quả
    print(f"\n[ KẾT QUẢ CHO ẢNH: {image_path} ]")
    print(f"--- Mô hình dự đoán Folder: {predicted_folder_name}")
    print(f"--- Mệnh giá: {denom}")
    print(f"--- Mô tả hình ảnh:")
    # Định dạng lại mô tả cho dễ đọc
    wrapped_desc = "\n    ".join(desc.split(". "))
    print(f"    {wrapped_desc}")
    print("="*60)

# --- 5. CHẠY TEST TRÊN 5 ẢNH ---
print("\nBắt đầu test trên 5 ảnh mới...\n")
for img_p in image_list:
    predict_and_show_info(img_p)