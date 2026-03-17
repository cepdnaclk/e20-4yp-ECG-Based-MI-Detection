"""
Dataset C Preprocessing: pure_normal vs all MI (certain + uncertain)
Loads ECG signals, applies filtering and normalization, saves as .npy arrays
"""

import sys
sys.path.append('../../shared_utils')

import numpy as np
import pandas as pd
from preprocessing import (
    load_ecg_batch, 
    remove_baseline_wander, 
    standardize_signals,
    verify_data_quality,
    compute_class_weights,
    save_preprocessing_info
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# ==========================================
# CONFIGURATION
# ==========================================

# Paths
PTBXL_PATH = '/scratch1/e20-fyp-ecg-ihd-detection/data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
CSV_DIR = '../../csvs/'
DATA_DIR = '../data/'

# CSV files
TRAIN_CSV = CSV_DIR + 'ptbxl_datasetC_train.csv'
# VAL_CSV = CSV_DIR + 'ptbxl_dataset_val.csv'
# TEST_CSV = CSV_DIR + 'ptbxl_dataset_test.csv'

VAL_CSV = CSV_DIR + 'ptbxl_dataset_val_certain_vs_normal.csv'
TEST_CSV = CSV_DIR + 'ptbxl_dataset_test_certain_vs_normal.csv'

print(f"\n{'='*60}")
print("Dataset C Preprocessing: pure_normal vs all MI")
print(f"{'='*60}")

# ==========================================
# 1. LOAD CSV FILES
# ==========================================

print("\n[Step 1] Loading CSV files...")

df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

print(f"✅ Train: {len(df_train)} samples")
print(f"✅ Val: {len(df_val)} samples")
print(f"✅ Test: {len(df_test)} samples")

# Print group distribution
print(f"\nTrain groups:")
print(df_train['main_3group'].value_counts())

print(f"\nVal groups:")
print(df_val['main_3group'].value_counts())

print(f"\nTest groups:")
print(df_test['main_3group'].value_counts())

# ==========================================
# 2. LOAD ECG SIGNALS
# ==========================================

print(f"\n{'='*60}")
print("[Step 2] Loading ECG signals from PTB-XL...")
print("This may take several minutes...")

X_train = load_ecg_batch(df_train, PTBXL_PATH, use_hr=False)
X_val = load_ecg_batch(df_val, PTBXL_PATH, use_hr=False)
X_test = load_ecg_batch(df_test, PTBXL_PATH, use_hr=False)

print(f"\n✅ Loaded signals:")
print(f"   Train: {X_train.shape}")
print(f"   Val: {X_val.shape}")
print(f"   Test: {X_test.shape}")

# ==========================================
# 3. EXTRACT LABELS
# ==========================================

print(f"\n{'='*60}")
print("[Step 3] Extracting labels...")

y_train = df_train['label_mi_vs_normal'].values
y_val = df_val['label_mi_vs_normal'].values
y_test = df_test['label_mi_vs_normal'].values

# Also save group labels for stratified evaluation later
groups_train = df_train['main_3group'].values
groups_val = df_val['main_3group'].values
groups_test = df_test['main_3group'].values

print(f"✅ Labels extracted")
print(f"\nTrain label distribution:")
print(f"   Normal (0): {(y_train==0).sum()} ({(y_train==0).mean()*100:.2f}%)")
print(f"   MI (1): {(y_train==1).sum()} ({(y_train==1).mean()*100:.2f}%)")

print(f"\nVal label distribution:")
print(f"   Normal (0): {(y_val==0).sum()} ({(y_val==0).mean()*100:.2f}%)")
print(f"   MI (1): {(y_val==1).sum()} ({(y_val==1).mean()*100:.2f}%)")

print(f"\nTest label distribution:")
print(f"   Normal (0): {(y_test==0).sum()} ({(y_test==0).mean()*100:.2f}%)")
print(f"   MI (1): {(y_test==1).sum()} ({(y_test==1).mean()*100:.2f}%)")

# ==========================================
# 4. REMOVE BASELINE WANDER
# ==========================================

print(f"\n{'='*60}")
print("[Step 4] Removing baseline wander (0.5 Hz high-pass filter)...")

X_train_filtered = remove_baseline_wander(X_train, sampling_rate=100, cutoff=0.5)
X_val_filtered = remove_baseline_wander(X_val, sampling_rate=100, cutoff=0.5)
X_test_filtered = remove_baseline_wander(X_test, sampling_rate=100, cutoff=0.5)

print("✅ Baseline wander removed")

# ==========================================
# 5. STANDARDIZATION (Lead-wise Z-score)
# ==========================================

print(f"\n{'='*60}")
print("[Step 5] Standardizing signals (lead-wise Z-score)...")

scaler_path = DATA_DIR + 'scaler.pkl'
X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_signals(
    X_train_filtered, X_val_filtered, X_test_filtered, scaler_path
)

print("✅ Signals standardized")

# ==========================================
# 6. DATA QUALITY VERIFICATION
# ==========================================

print(f"\n{'='*60}")
print("[Step 6] Verifying data quality...")

verify_data_quality(X_train_scaled, y_train, "Train")
verify_data_quality(X_val_scaled, y_val, "Validation")
verify_data_quality(X_test_scaled, y_test, "Test")

# ==========================================
# 7. COMPUTE CLASS WEIGHTS
# ==========================================

print(f"\n{'='*60}")
print("[Step 7] Computing class weights for loss function...")

class_weights = compute_class_weights(y_train)

# ==========================================
# 8. SAVE PREPROCESSED DATA
# ==========================================

print(f"\n{'='*60}")
print("[Step 8] Saving preprocessed data...")

# Save signals
np.save(DATA_DIR + 'X_train.npy', X_train_scaled)
np.save(DATA_DIR + 'X_val.npy', X_val_scaled)
np.save(DATA_DIR + 'X_test.npy', X_test_scaled)

# Save labels
np.save(DATA_DIR + 'y_train.npy', y_train)
np.save(DATA_DIR + 'y_val.npy', y_val)
np.save(DATA_DIR + 'y_test.npy', y_test)

# Save group labels
np.save(DATA_DIR + 'groups_train.npy', groups_train)
np.save(DATA_DIR + 'groups_val.npy', groups_val)
np.save(DATA_DIR + 'groups_test.npy', groups_test)

print(f"\n✅ All data saved to {DATA_DIR}")
print(f"\nFiles created:")
print(f"   - X_train.npy ({X_train_scaled.nbytes / 1e6:.2f} MB)")
print(f"   - X_val.npy ({X_val_scaled.nbytes / 1e6:.2f} MB)")
print(f"   - X_test.npy ({X_test_scaled.nbytes / 1e6:.2f} MB)")
print(f"   - y_train.npy, y_val.npy, y_test.npy")
print(f"   - groups_train.npy, groups_val.npy, groups_test.npy")
print(f"   - scaler.pkl")

# ==========================================
# 9. SAVE PREPROCESSING METADATA
# ==========================================

save_preprocessing_info(
    DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV,
    len(y_train), len(y_val), len(y_test), class_weights
)

# ==========================================
# SUMMARY
# ==========================================

print(f"\n{'='*60}")
print("PREPROCESSING COMPLETE!")
print(f"{'='*60}")
print(f"\nDataset C Summary:")
print(f"  Training: {X_train_scaled.shape[0]} samples")
print(f"    - Groups in train: {np.unique(groups_train)}")
print(f"  Validation: {X_val_scaled.shape[0]} samples")
print(f"    - Groups in val: {np.unique(groups_val)}")
print(f"  Test: {X_test_scaled.shape[0]} samples")
print(f"    - Groups in test: {np.unique(groups_test)}")
print(f"\nSignal shape: (1000 timesteps, 12 leads)")
print(f"Sampling rate: 100 Hz")
print(f"Preprocessing: Baseline wander removal + Lead-wise normalization")
print(f"\nClass weights for training:")
print(f"  Normal (0): {class_weights[0]:.3f}")
print(f"  MI (1): {class_weights[1]:.3f}")
print(f"\n{'='*60}")
print("Next step: Run 2_train.py to train the CNN-LSTM model")
print(f"{'='*60}\n")