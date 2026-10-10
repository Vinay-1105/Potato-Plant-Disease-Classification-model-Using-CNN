# Training Pipeline

This directory contains scripts to audit the dataset, split the data, train the baseline CNN, evaluate model performance, and analyze confidence thresholds.

---

## Scripts

### 1. `01_split_and_phash.py`
- **What it does**: Audits class counts and imbalance ratio across `dataset/PlantVillage`, performs a stratified 80/10/10 train/validation/test split with `random_state=42`, verifies disjoint sets across splits, generates `training/splits.json`, and computes 64-bit perceptual image hashes (`phash`) to flag potential train/test near-duplicates (Hamming distance $\le 4$).
- **How to run**:
  ```bash
  python training/01_split_and_phash.py
  ```
- **Provenance**: Original logic recovered from the execution log; edits were path-only (repo-relative paths, separator normalization, default output name of 02_train.py changed to model_retrained.keras).

---

### 2. `02_train.py`
- **What it does**: Loads train and validation splits from `training/splits.json`, computes balanced class weights on the training set, defines the 4-block CNN architecture (Conv2D + MaxPooling2D layers, Dense 64, Dense 3 with softmax), and trains with EarlyStopping (`patience=7`, restoring best weights) and ReduceLROnPlateau. Saves training metrics to `logs/history_retrained.json` and outputs model weights to `logs/model_retrained.keras` (never overwriting committed files).
- **How to run**:
  ```bash
  python training/02_train.py
  ```
- **Provenance**: Original logic recovered from the execution log; edits were path-only (repo-relative paths, separator normalization, default outputs redirected to `logs/` as `logs/model_retrained.keras` and `logs/history_retrained.json`).

---

### 3. `03_evaluate.py`
- **What it does**: Loads the 216 test set images from `training/splits.json` and the trained `model_v2.keras`, computes overall test accuracy (93.06%), prints the classification report and confusion matrix, and generates visual evaluation assets (`confusion_matrix.png`, `training_curves.png`, and `prediction_samples.png`) in `assets/`.
- **How to run**:
  ```bash
  python training/03_evaluate.py
  ```
- **Provenance**: Original logic recovered from the execution log; edits were path-only (repo-relative paths, separator normalization, default output name of 02_train.py changed to model_retrained.keras).

---

### 4. `04_threshold_analysis.py`
- **What it does**: Evaluates prediction accuracy and coverage across confidence thresholds (0.6, 0.7, 0.8, 0.9) on both the validation set (215 images) and the test set (216 images) using `model_v2.keras`.
- **How to run**:
  ```bash
  python training/04_threshold_analysis.py
  ```
- **Provenance**: Reconstructed from scratch/verify_step2_step4.py, not the original script. It reproduces the original validation and test threshold numbers exactly.

---

## Verification Note

01_split_and_phash.py rewrites training/splits.json with the same split (identical membership and order); only the JSON formatting differs.
Verified: 01_split_and_phash.py regenerates a split with identical membership and order to splits.json (JSON layout differs only). Production requirements (requirements.txt) were verified in a clean environment. Dev requirements were installed on top of that environment, not in a second clean environment.
