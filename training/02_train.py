import os
import sys
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits.json")
LOGS_DIR = os.path.join(REPO_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
OUTPUT_MODEL = os.path.join(LOGS_DIR, "model_retrained.keras")
HISTORY_FILE = os.path.join(LOGS_DIR, "history_retrained.json")

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
print(f"Target class order: {class_names}")

with open(SPLITS_FILE, "r") as f:
    splits = json.load(f)

train_files = [os.path.normpath(f.replace("\\", "/")) for f in splits["train_files"]]
val_files = [os.path.normpath(f.replace("\\", "/")) for f in splits["val_files"]]
test_files = [os.path.normpath(f.replace("\\", "/")) for f in splits["test_files"]]

def get_label(path):
    for idx, cname in enumerate(class_names):
        if cname in path:
            return idx
    raise ValueError(f"Unknown class for path {path}")

train_y = [get_label(f) for f in train_files]
val_y = [get_label(f) for f in val_files]
test_y = [get_label(f) for f in test_files]

# Compute balanced class weights on training set only
unique_classes = np.unique(train_y)
weights = compute_class_weight('balanced', classes=unique_classes, y=train_y)
class_weights_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
print(f"Computed Class Weights: {class_weights_dict}")

BATCH_SIZE = 32
IMAGE_SIZE = 256

def load_img(path, label):
    # Convert relative path to full path if needed
    full_path = tf.strings.join([REPO_DIR, tf.constant(os.sep), path])
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label

def create_dataset(paths, labels, is_train=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if is_train:
        ds = ds.shuffle(buffer_size=1000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_files, train_y, is_train=True)
val_ds = create_dataset(val_files, val_y, is_train=False)
test_ds = create_dataset(test_files, test_y, is_train=False)

print(f"Datasets built successfully. Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")

def build_baseline_model():
    preprocessing = keras.Sequential([
        keras.layers.Resizing(height=IMAGE_SIZE, width=IMAGE_SIZE),
        keras.layers.Rescaling(scale=1./255),
        keras.layers.RandomFlip(mode='horizontal_and_vertical'),
        keras.layers.RandomRotation(factor=0.2)
    ], name="preprocessing")

    model = keras.Sequential([
        preprocessing,
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ], name="baseline_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_baseline_model()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath=OUTPUT_MODEL, monitor='val_loss', save_best_only=True, verbose=1)
]

print("\n--- BASELINE CNN TRAINING (up to 50 epochs) ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=2
)

hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_FILE, "w") as f:
    json.dump(hist_data, f, indent=2)

print("\n--- TRAINING FINISHED ---")
best_epoch = np.argmin(hist_data['val_loss']) + 1
print(f"Best Val Loss achieved at epoch {best_epoch}:")
print(f"  Train Acc: {hist_data['accuracy'][best_epoch-1]*100:.2f}%")
print(f"  Val Acc: {hist_data['val_accuracy'][best_epoch-1]*100:.2f}%")
print(f"  Train Loss: {hist_data['loss'][best_epoch-1]:.4f}")
print(f"  Val Loss: {hist_data['val_loss'][best_epoch-1]:.4f}")
