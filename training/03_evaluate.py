import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits.json")
MODEL_PATH = os.path.join(REPO_DIR, "model_v2.keras")
ASSETS_DIR = os.path.join(REPO_DIR, "assets")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
os.makedirs(ASSETS_DIR, exist_ok=True)

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load test split
with open(SPLITS_FILE, "r") as f:
    splits = json.load(f)
test_files = [os.path.normpath(f.replace("\\", "/")) for f in splits["test_files"]]

# Determine labels from file paths
test_labels = []
for f in test_files:
    for idx, cname in enumerate(class_names):
        if cname in f:
            test_labels.append(idx)
            break
test_labels = np.array(test_labels)

# Build test dataset
IMAGE_SIZE = 256
BATCH_SIZE = 32

def load_img(path, label):
    full_path = tf.strings.join([REPO_DIR, tf.constant(os.sep), path])
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label

test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_ds = test_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Final evaluation on test set
print("\n--- FINAL EVALUATION ON TEST SET (216 images) ---")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}% ({int(round(test_acc * len(test_labels)))} / {len(test_labels)})")

# Generate predictions and probabilities
y_probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_probs, axis=1)
confidences = np.max(y_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix:")
print("Predicted ->", class_names)
for idx, row in enumerate(cm):
    print(f"True {class_names[idx]}: {row}")

# Classification Report
report = classification_report(test_labels, y_pred, target_names=class_names, digits=4, output_dict=True)
print("\nClassification Report:")
print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
for cname in class_names:
    metrics = report[cname]
    print(f"{cname:<25} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}")

macro = report['macro avg']
print(f"{'Macro Average':<25} {macro['precision']:<10.4f} {macro['recall']:<10.4f} {macro['f1-score']:<10.4f} {macro['support']:<10}")

# Calibration Check
correct_mask = (y_pred == test_labels)
wrong_mask = ~correct_mask

avg_conf_correct = np.mean(confidences[correct_mask]) if np.sum(correct_mask) > 0 else 0.0
avg_conf_wrong = np.mean(confidences[wrong_mask]) if np.sum(wrong_mask) > 0 else 0.0
wrong_high_conf_count = np.sum((confidences > 0.90) & wrong_mask)

print("\n--- CALIBRATION CHECK ---")
print(f"Total Correct Predictions: {np.sum(correct_mask)} / {len(test_labels)} ({np.mean(correct_mask)*100:.2f}%)")
print(f"Total Wrong Predictions: {np.sum(wrong_mask)} / {len(test_labels)}")
print(f"Average Confidence on CORRECT predictions: {avg_conf_correct*100:.2f}%")
print(f"Average Confidence on WRONG predictions: {avg_conf_wrong*100:.2f}%" if np.sum(wrong_mask) > 0 else "Average Confidence on WRONG predictions: N/A")
print(f"Wrong predictions with confidence > 90%: {wrong_high_conf_count}")

# Generate assets if history exists
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    epochs_range = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    plt.axvline(x=18, color='green', linestyle='--', label='Best Weights (Epoch 18)')
    plt.title('Training vs Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.axvline(x=18, color='green', linestyle='--', label='Best Weights (Epoch 18)')
    plt.title('Training vs Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout()
    curves_path = os.path.join(ASSETS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

# Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix on Test Set', fontsize=12, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(len(class_names))
short_names = ['Early Blight', 'Late Blight', 'Healthy']
plt.xticks(tick_marks, short_names, rotation=30)
plt.yticks(tick_marks, short_names)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
cm_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"\nAssets verified in {ASSETS_DIR}")
