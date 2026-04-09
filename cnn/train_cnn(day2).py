import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 1. CẤU HÌNH ĐƯỜNG DẪN
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")
mapping_path = "app/mapping.json"
model_save_path = "cnn/cnn_model.h5" # Lưu theo yêu cầu checklist

# Tạo thư mục cnn nếu chưa có
if not os.path.exists('cnn'):
    os.makedirs('cnn')

# Load mapping
with open(mapping_path, 'r', encoding='utf-8') as f:
    mapping = json.load(f)

def get_denomination_from_folder(folder_name):
    if folder_name == 'noise': return 'Nhiễu/Khác'
    return mapping.get(folder_name, {}).get('denomination', folder_name)

# 2. DATA AUGMENTATION (Nâng cấp: Rotate, Blur, Brightness)
IMG_SIZE = 224
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=15),                   # Thêm Xoay
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # Thêm Độ sáng/Tương phản
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),      # Thêm Blur
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. DATALOADER
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)

# 4. MÔ HÌNH (Nâng cấp: Fine-tuning để Improve Accuracy)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)

# Đóng băng các layer đầu, nhưng mở layer3 và layer4 để học sâu hơn về tiền VN
for name, child in model.named_children():
    if name in ['layer3', 'layer4', 'fc']:
        for param in child.parameters():
            param.requires_grad = True
    else:
        for param in child.parameters():
            param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss & Optimizer (Sử dụng LR nhỏ để fine-tune mượt hơn)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# 5. TRAINING LOOP
num_epochs = 20 # Bạn có thể tăng lên nếu cần
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
    
    train_acc = train_correct.double() / len(train_dataset)
    
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
    
    val_acc = val_correct.double() / len(val_dataset)
    val_loss = val_loss / len(val_dataset)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"  --> Saved better model to {model_save_path}")

# 6. FINAL EVALUATION
print("\n--- Đang đánh giá trên tập Test ---")
model.load_state_dict(torch.load(model_save_path))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

class_names = [get_denomination_from_folder(cls) for cls in train_dataset.classes]
print(classification_report(y_true, y_pred, target_names=class_names))