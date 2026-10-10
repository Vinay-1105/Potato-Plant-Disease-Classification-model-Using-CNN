import os
import glob
import random
import json
import numpy as np
import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_DIR, "dataset", "PlantVillage")
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits.json")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Class names: {class_names}")

# Step 1: Data Check
class_counts = {}
all_filepaths = []
all_labels = []

for idx, cname in enumerate(class_names):
    c_path = os.path.join(DATA_DIR, cname)
    files = sorted([
        os.path.join("dataset", "PlantVillage", cname, f) 
        for f in os.listdir(c_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    class_counts[cname] = len(files)
    all_filepaths.extend(files)
    all_labels.extend([idx] * len(files))

total_images = sum(class_counts.values())
print("\n--- STEP 1: DATA CHECK ---")
for cname, count in class_counts.items():
    print(f"  {cname}: {count} images ({count/total_images*100:.2f}%)")
print(f"Total images: {total_images}")

min_count = min(class_counts.values())
max_count = max(class_counts.values())
imbalance_ratio = max_count / min_count
print(f"Max class count: {max_count}, Min class count: {min_count}, Ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 1.5:
    print(f"FLAGGED: Class imbalance is {imbalance_ratio:.2f}x (> 1.5x threshold)!")
else:
    print("Class balance is within acceptable limit (<= 1.5x).")

# Step 2: Leak-Free Split
print("\n--- STEP 2: LEAK-FREE SPLIT ---")
# Stratified 80/10/10 split
train_files, val_test_files, train_y, val_test_y = train_test_split(
    all_filepaths, all_labels, test_size=0.20, random_state=SEED, stratify=all_labels
)
val_files, test_files, val_y, test_y = train_test_split(
    val_test_files, val_test_y, test_size=0.50, random_state=SEED, stratify=val_test_y
)

print(f"Train size: {len(train_files)} (80%)")
print(f"Validation size: {len(val_files)} (10%)")
print(f"Test size: {len(test_files)} (10%)")

# Assertion
assert len(set(train_files).intersection(set(val_files))) == 0, "Train and Val share files!"
assert len(set(train_files).intersection(set(test_files))) == 0, "Train and Test share files!"
assert len(set(val_files).intersection(set(test_files))) == 0, "Val and Test share files!"
print("Assertion Passed: Train, Validation, and Test share zero file paths!")

# Save file split lists
with open(SPLITS_FILE, "w") as f:
    json.dump({
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files
    }, f, indent=2)
print(f"Splits saved to {SPLITS_FILE}")

# Step 2b: Near-duplicate check
print("\n--- STEP 2b: NEAR-DUPLICATE CHECK ---")
print("Computing perceptual hashes (phash) for all images...")
train_hashes = []
for f in train_files:
    abs_f = os.path.join(REPO_DIR, f)
    try:
        h = imagehash.phash(Image.open(abs_f))
        train_hashes.append(h)
    except Exception as e:
        pass

test_near_duplicates = 0
duplicate_matches = []
HASH_DIST_THRESH = 4

for idx, f in enumerate(test_files):
    abs_f = os.path.join(REPO_DIR, f)
    try:
        th = imagehash.phash(Image.open(abs_f))
        min_dist = min([th - h for h in train_hashes])
        if min_dist <= HASH_DIST_THRESH:
            test_near_duplicates += 1
            duplicate_matches.append((f, min_dist))
    except Exception as e:
        pass

dup_percent = (test_near_duplicates / len(test_files)) * 100
print(f"Total test images: {len(test_files)}")
print(f"Test images with a near-duplicate in training set (Hamming dist <= {HASH_DIST_THRESH}): {test_near_duplicates} ({dup_percent:.2f}%)")

if dup_percent > 2.0:
    print(f"FLAGGED: Duplicate percentage {dup_percent:.2f}% exceeds 2% threshold!")
else:
    print("Near-duplicate percentage is within 2% limit.")
