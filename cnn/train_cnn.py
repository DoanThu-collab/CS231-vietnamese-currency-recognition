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

# -------------------- 1. ĐƯỜNG DẪN --------------------
data_dir = "data"  # hoặc "data/processed" nếu bạn để trong đó
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")
mapping_path = "app/mapping.json"  # vẫn giữ để lấy mô tả, denomination

# Load mapping để lấy thông tin mệnh giá và mô tả
with open(mapping_path, 'r', encoding='utf-8') as f:
    mapping = json.load(f)

# Lấy danh sách các lớp (tên thư mục con trong train)
classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
classes.sort()
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
num_classes = len(classes)

# -------------------- 2. TRANSFORMS --------------------
IMG_SIZE = 224
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # vẫn resize để an toàn (ảnh đã resize nhưng giữ)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- 3. TẠO DATALOADER --------------------
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

# Gán lại nhãn cho phù hợp với mapping (vì ImageFolder tự gán nhãn theo alphabet)
# Nhưng nếu tên thư mục con giống với key trong mapping thì không sao.
# Tuy nhiên, để lấy denomination và description, ta cần ánh xạ từ tên folder.
# Ta sẽ tạo một hàm helper:

def get_denomination_from_folder(folder_name):
    if folder_name == 'noise':
        return 'Nhiễu/Khác'
    return mapping.get(folder_name, {}).get('denomination', 'Unknown')

def get_description_from_folder(folder_name):
    if folder_name == 'noise':
        return 'Hình ảnh Representative của nhiễu hoặc các lớp khác'
    return mapping.get(folder_name, {}).get('description', '')

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# -------------------- 4. MÔ HÌNH --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# -------------------- 5. HUẤN LUYỆN --------------------
num_epochs = 30
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
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
    train_loss = train_loss / len(train_dataset)
    
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
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("  -> Saved best model")

# -------------------- 6. ĐÁNH GIÁ TEST --------------------
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Lấy tên các lớp theo đúng thứ tự của ImageFolder
class_names = [cls for cls, _ in sorted(train_dataset.class_to_idx.items(), key=lambda x: x[1])]
target_names = [get_denomination_from_folder(cls) for cls in class_names]

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()