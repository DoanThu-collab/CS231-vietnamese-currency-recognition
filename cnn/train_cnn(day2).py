import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm

# 1. CẤU HÌNH & ĐƯỜNG DẪN
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")
mapping_path = "app/mapping.json"
model_save_path = "cnn/cnn_model.h5"

if not os.path.exists('cnn'): os.makedirs('cnn')

# Load mapping để hiển thị tên mệnh giá chuẩn
with open(mapping_path, 'r', encoding='utf-8') as f:
    mapping = json.load(f)

def get_name(folder_name):
    if folder_name == 'noise': return 'Nhiễu/Khác'
    return mapping.get(folder_name, {}).get('denomination', folder_name)

# 2. ANTI-OVERFITTING AUGMENTATION (Ép mô hình học khó hơn)
IMG_SIZE = 224
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)), # Biến dạng hình học
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),                   # Thay đổi góc chụp
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),       # Ánh sáng gắt
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),              # Làm mờ ảnh
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. DATALOADER
train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_ds = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 4. MODEL CẢI TIẾN (Thêm Dropout & Fine-tuning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights='IMAGENET1K_V1')

# Mở khóa layer 3 và 4 để fine-tune đặc trưng tiền Việt
for name, param in model.named_parameters():
    param.requires_grad = any(x in name for x in ['layer3', 'layer4', 'fc'])

# Thay thế FC bằng Sequential có Dropout để chống Overfit
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, len(train_ds.classes))
)
model = model.to(device)

# Optimizer với Weight Decay (L2 Regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# 5. TRAINING VỚI EARLY STOPPING
num_epochs = 50
best_val_loss = float('inf')
patience = 7  # Dừng nếu 7 epoch liên tiếp loss không giảm
counter = 0

print(f"Bắt đầu huấn luyện trên {device}...")

for epoch in range(num_epochs):
    model.train()
    t_loss, t_corr = 0.0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(out, 1)
        t_corr += torch.sum(preds == lbls.data)

    model.eval()
    v_loss, v_corr = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            v_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            v_corr += torch.sum(preds == lbls.data)

    avg_v_loss = v_loss / len(val_ds)
    avg_v_acc = v_corr.double() / len(val_ds)
    scheduler.step(avg_v_loss)

    print(f"Val Acc: {avg_v_acc:.4f} | Val Loss: {avg_v_loss:.4f}")

    # Early Stopping Check (Dựa trên Loss thay vì Acc để chống ảo)
    if avg_v_loss < best_val_loss:
        best_val_loss = avg_v_loss
        torch.save(model.state_dict(), model_save_path)
        counter = 0
        print(f"  --> Saved model to {model_save_path}")
    else:
        counter += 1
        if counter >= patience:
            print("Early Stopping! Model đã đạt ngưỡng tổng quát hóa tốt nhất.")
            break

# 6. KIỂM TRA CUỐI CÙNG
model.load_state_dict(torch.load(model_save_path))
model.eval()
y_t, y_p = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        out = model(imgs.to(device))
        _, preds = torch.max(out, 1)
        y_t.extend(lbls.numpy())
        y_p.extend(preds.cpu().numpy())

target_names = [get_name(c) for c in train_ds.classes]
print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(y_t, y_p, target_names=target_names))