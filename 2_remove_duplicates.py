# remove_duplicates.py
import hashlib, os

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def build_map(folder):
    d = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, f)
                d.setdefault(file_hash(path), []).append(path)
    return d

train_map = build_map("dataset_split/train")
val_map   = build_map("dataset_split/val")
test_map  = build_map("dataset_split/test")

def remove_duplicates(source_map, target_map, target_folder):
    dup_hashes = set(source_map.keys()) & set(target_map.keys())
    print(f"Removing {len(dup_hashes)} duplicates from {target_folder}")
    for h in dup_hashes:
        for path in target_map[h]:
            os.remove(path)
            print(f"Removed {path}")

remove_duplicates(train_map, val_map, "val")
remove_duplicates(train_map, test_map, "test")
remove_duplicates(val_map, test_map, "test")

print("Duplicate removal complete.")
