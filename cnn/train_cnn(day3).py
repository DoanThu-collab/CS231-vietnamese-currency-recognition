import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thư mục đích để lưu kết quả
output_dir = "cnn" 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, "cnn_model.h5") # Đường dẫn file model
mapping_path = "app/mapping.json"
data_dir = "data/val" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Mapping
with open(mapping_path, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
class_names = sorted(list(mapping.keys()))
if 'noise' not in class_names: class_names.append('noise')

# 2. Chuẩn bị Dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. Load Model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, len(class_names)))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 4. Dự đoán
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in loader:
        outputs = model(imgs.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 5. VẼ VÀ LƯU HEATMAP VÀO THƯ MỤC CNN
display_names = [mapping.get(c, {}).get('denomination', c) for c in class_names]
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_names, yticklabels=display_names)
plt.title('Confusion Matrix')

# Lưu trực tiếp vào thư mục cnn
save_path_cm = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(save_path_cm, dpi=300, bbox_inches='tight')
print(f"✅ Đã lưu Confusion Matrix vào: {save_path_cm}")

# 6. LƯU REPORT VÀO THƯ MỤC CNN
report = classification_report(all_labels, all_preds, target_names=display_names)
save_path_txt = os.path.join(output_dir, 'classification_report.txt')
with open(save_path_txt, "w", encoding="utf-8") as f:
    f.write(report)

print(f"✅ Đã lưu Classification Report vào: {save_path_txt}")
print("\n=== NỘI DUNG REPORT ===")
print(report)