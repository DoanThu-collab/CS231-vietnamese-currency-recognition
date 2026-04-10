import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import cv2
import numpy as np

# --- 1. CẤU HÌNH ---
output_dir = "cnn"
model_path = os.path.join(output_dir, "cnn_model.h5")
mapping_path = "app/mapping.json"
output_test_dir = os.path.join(output_dir, "test_results_bbox")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(output_test_dir):
    os.makedirs(output_test_dir)

# Danh sách ảnh test
image_list = ["VNĐ1.webp", "VNĐ2.webp", "VNĐ3.jpg", "VNĐ4.webp", "VNĐ5.webp"]

# --- 2. LOAD MAPPING & MODEL ---
try:
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
except Exception as e:
    print(f"❌ Lỗi đọc file mapping: {e}")
    exit()

class_names = sorted(list(mapping.keys()))
if 'noise' not in class_names: class_names.append('noise')

print(f"Đang tải mô hình từ: {model_path}...")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, len(class_names))
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_on_crop(pil_image_crop):
    img_t = preprocess(pil_image_crop)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        out = model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, index = torch.max(prob, 1)
    folder = class_names[index[0]]
    denom = mapping.get(folder, {}).get('denomination', 'Nhiễu')
    return denom, conf.item()

def detect_and_predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy: {image_path}")
        return

    # Đọc ảnh an toàn với đường dẫn tiếng Việt
    img_array = np.fromfile(image_path, np.uint8)
    cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if cv_img is None: 
        print(f"❌ Không thể giải mã ảnh: {image_path}")
        return
    
    draw_img = cv_img.copy()
    h_orig, w_orig = cv_img.shape[:2]
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blurred, 30, 100) 
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h_orig * w_orig * 0.03): continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h > 0 else 0
        
        if 1.2 < aspect_ratio < 4.0 or 0.25 < aspect_ratio < 0.8:
            detected_count += 1
            crop_cv = cv_img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            denom, conf = predict_on_crop(crop_pil)
            
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"{denom} {conf*100:.1f}%"
            cv2.putText(draw_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            print(f"   -> Tìm thấy {denom} tại {x},{y}")

    if detected_count == 0:
        cv2.putText(draw_img, "No currency detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # --- LƯU VÀ HIỂN THỊ ẢNH ---
    out_path = os.path.join(output_test_dir, "res_" + os.path.basename(image_path))
    
    ext = os.path.splitext(out_path)[1]
    if not ext: ext = ".jpg"
    is_success, im_buf_arr = cv2.imencode(ext, draw_img)
    if is_success:
        im_buf_arr.tofile(out_path)
        print(f"✅ Đã lưu: {out_path}")
    else:
        print(f"❌ Lỗi không thể lưu ảnh: {out_path}")

    # Hiển thị ảnh (Nhấn phím bất kỳ để xem ảnh tiếp theo)
    window_name = f"Result - {os.path.basename(image_path)}"
    cv2.imshow(window_name, draw_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# --- CHẠY ---
current_dir = os.path.dirname(os.path.abspath(__file__))

for img in image_list:
    full_image_path = os.path.join(current_dir, img)
    detect_and_predict(full_image_path)