import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits.json")
MODEL_PATH = os.path.join(REPO_DIR, "model_v2.keras")

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def get_class_idx(path):
    if "Potato___Early_blight" in path:
        return 0
    elif "Potato___Late_blight" in path:
        return 1
    elif "Potato___healthy" in path:
        return 2
    raise ValueError(f"Unknown class for path {path}")

with open(SPLITS_FILE, "r") as f:
    splits = json.load(f)

val_files = [os.path.normpath(f.replace("\\", "/")) for f in splits["val_files"]]
test_files = [os.path.normpath(f.replace("\\", "/")) for f in splits["test_files"]]

model = keras.models.load_model(MODEL_PATH)
print("Loaded model_v2.keras successfully.")

# 1. Validation Set Threshold Evaluation (215 images)
print("\n=== VALIDATION SET THRESHOLD EVALUATION (215 images) ===")
val_preds_list = []
for rel_path in val_files:
    true_cls = get_class_idx(rel_path)
    abs_path = os.path.join(REPO_DIR, rel_path)
    img = Image.open(abs_path).convert('RGB').resize((256, 256))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0) # Raw [0, 255]
    preds = model.predict(arr, verbose=0)[0]
    pred_cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    val_preds_list.append((true_cls, pred_cls, conf))

thresholds = [0.6, 0.7, 0.8, 0.9]
total_val = len(val_preds_list)

print(f"{'Threshold':<10} | {'Coverage':<12} | {'Accepted':<10} | {'Accuracy':<12} | {'Errors':<10} | {'Rejected':<10}")
print("-" * 75)
for t in thresholds:
    accepted = [p for p in val_preds_list if p[2] >= t]
    rejected_count = total_val - len(accepted)
    cov = len(accepted) / total_val
    if len(accepted) > 0:
        correct = sum(1 for p in accepted if p[0] == p[1])
        acc = correct / len(accepted)
        errors = len(accepted) - correct
    else:
        acc = 0.0
        errors = 0
    print(f"{t:<10.1f} | {cov*100:<11.2f}% | {len(accepted):<10} | {acc*100:<11.2f}% | {errors:<10} | {rejected_count:<10}")

chosen_threshold = 0.7
print(f"\nRecommended threshold: {chosen_threshold}")

# 2. Test Set Evaluation across thresholds (216 images)
print(f"\n=== TEST SET EVALUATION ACROSS THRESHOLDS (216 images) ===")
test_preds_list = []
for rel_path in test_files:
    true_cls = get_class_idx(rel_path)
    abs_path = os.path.join(REPO_DIR, rel_path)
    img = Image.open(abs_path).convert('RGB').resize((256, 256))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    pred_cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    test_preds_list.append((true_cls, pred_cls, conf))

total_test = len(test_preds_list)
print(f"{'Threshold':<10} | {'Coverage':<12} | {'Accepted':<10} | {'Accuracy':<12} | {'Errors':<10} | {'Rejected':<10}")
print("-" * 75)
for t in thresholds:
    accepted = [p for p in test_preds_list if p[2] >= t]
    rejected_count = total_test - len(accepted)
    cov = len(accepted) / total_test
    if len(accepted) > 0:
        correct = sum(1 for p in accepted if p[0] == p[1])
        acc = correct / len(accepted)
        errors = len(accepted) - correct
    else:
        acc = 0.0
        errors = 0
    print(f"{t:<10.1f} | {cov*100:<11.2f}% | {len(accepted):<10} | {acc*100:<11.2f}% | {errors:<10} | {rejected_count:<10}")
