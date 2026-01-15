import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ---------- Config ----------
DATA_DIR = "dataset_split"   # dataset folder (train/val/test)
MODEL_NAME = "vit_tiny_patch16_224"  # smaller ViT to save memory
IMG_SIZE = 224
BATCH_SIZE = 1  # smaller batch size for low GPU
LR = 2e-5
EPOCHS = 12
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
SEED = 42
# ----------------------------

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.cuda.empty_cache()

# ---------- Transforms ----------
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

# ---------- Custom Dataset ----------
from torch.utils.data import Dataset
class AlbumentationsImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.imgs = []
        self.targets = []
        classes = sorted(os.listdir(root))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for cls in classes:
            cls_folder = os.path.join(root, cls)
            if not os.path.isdir(cls_folder):
                continue
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    self.imgs.append(os.path.join(cls_folder, fname))
                    self.targets.append(self.class_to_idx[cls])
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        return image, target

# ---------- Load Data ----------
train_ds = AlbumentationsImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds = AlbumentationsImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
test_ds = AlbumentationsImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# ---------- Model ----------
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
model.to(DEVICE)

# ---------- Loss, Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# ---------- AMP Scaler ----------
scaler = torch.cuda.amp.GradScaler()

# ---------- Training ----------
best_val_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}] Train")
    for imgs, targets in loop:
        imgs, targets = imgs.to(DEVICE, dtype=torch.float), targets.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # mixed precision
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_ds)
    epoch_acc = accuracy_score(all_targets, all_preds)
    print(f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    # ---------- Validation ----------
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(DEVICE, dtype=torch.float), targets.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            val_preds.extend(preds.tolist())
            val_targets.extend(targets.detach().cpu().numpy().tolist())

    val_acc = accuracy_score(val_targets, val_preds)
    print(f"Val Acc: {val_acc:.4f}")
    scheduler.step(val_acc)

    # ---------- Save best ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': train_ds.class_to_idx,
        }, ckpt_path)
        print(f"Saved best model to {ckpt_path}")

# ---------- Test Evaluation ----------
print("Evaluating on test set...")
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs, targets = imgs.to(DEVICE, dtype=torch.float), targets.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        test_preds.extend(preds.tolist())
        test_targets.extend(targets.detach().cpu().numpy().tolist())

print("Test Acc:", accuracy_score(test_targets, test_preds))
print("Classification Report:")
print(classification_report(test_targets, test_preds, target_names=train_ds.classes))

# ---------- Save final model ----------
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_vit.pth"))
print("Training complete.")
