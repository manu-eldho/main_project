import hashlib, os

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def build_hash_map(folder):
    d = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg")):
                p = os.path.join(root, f)
                d.setdefault(file_hash(p), []).append(p)
    return d

train_map = build_hash_map("dataset_split/train")
val_map   = build_hash_map("dataset_split/val")
test_map  = build_hash_map("dataset_split/test")

# intersection
common = set(train_map.keys()) & set(val_map.keys()) & set(test_map.keys())
print("Hashes present in all three splits:", len(common))
print("Train/Val common:", len(set(train_map.keys()) & set(val_map.keys())))
print("Train/Test common:", len(set(train_map.keys()) & set(test_map.keys())))
print("Val/Test common:", len(set(val_map.keys()) & set(test_map.keys())))
