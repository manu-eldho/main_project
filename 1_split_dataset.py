import shutil
import random
from pathlib import Path

DATASET_DIR = Path("/home/final project/dataset/plantvillage")   
OUTPUT_DIR = Path("/home/final project/dataset_split")

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)
# Get all class folders (two levels deep)
classes = [d for d in DATASET_DIR.iterdir() if d.is_dir()]

if len(classes) == 1 and list(classes[0].iterdir()):
    DATASET_DIR = classes[0]  
    classes = [d for d in DATASET_DIR.iterdir() if d.is_dir()]

print(f"Classes found: {[cls.name for cls in classes]}")

for cls in classes:
    images = [f for f in cls.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    print(f"{cls.name}: {len(images)} images found")
    
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]
    
    for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        split_dir = OUTPUT_DIR / split_name / cls.name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_imgs:
            shutil.copy(img_path, split_dir / img_path.name)

print(f" Dataset split complete! Saved in: {OUTPUT_DIR}")
