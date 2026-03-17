"""
Shared Preprocessing Utilities for PTB-XL ECG Dataset
Handles signal loading, filtering, and normalization
"""

import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_ecg_signal(filename, ptbxl_path):
    """
    Load a single ECG signal from PTB-XL dataset
    
    Args:
        filename: Path to ECG file (e.g., 'records100/00000/00001_lr')
        ptbxl_path: Root path to PTB-XL dataset
    
    Returns:
        ECG signal array (1000 timesteps × 12 leads)
    """
    try:
        record = wfdb.rdsamp(ptbxl_path + filename)
        ecg_signal = record[0]  # Extract signal data
        return ecg_signal
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def load_ecg_batch(df, ptbxl_path, use_hr=False):
    """
    Load multiple ECG signals from dataframe
    
    Args:
        df: DataFrame with filename_lr or filename_hr columns
        ptbxl_path: Root path to PTB-XL dataset
        use_hr: Use high-res (500 Hz) if True, else low-res (100 Hz)
    
    Returns:
        Array of ECG signals (n_samples, 1000, 12)
    """
    filename_col = 'filename_hr' if use_hr else 'filename_lr'
    
    ecg_signals = []
    failed_indices = []
    
    for idx, filename in enumerate(df[filename_col].values):
        ecg = load_ecg_signal(filename, ptbxl_path)
        if ecg is not None:
            ecg_signals.append(ecg)
        else:
            failed_indices.append(idx)
    
    if failed_indices:
        print(f"Warning: Failed to load {len(failed_indices)} ECG signals")
    
    return np.array(ecg_signals)


def remove_baseline_wander(ecg_signal, sampling_rate=100, cutoff=0.5):
    """
    Remove baseline wander using high-pass Butterworth filter
    
    Args:
        ecg_signal: ECG signal array (timesteps, leads) or (n_samples, timesteps, leads)
        sampling_rate: Sampling frequency in Hz
        cutoff: High-pass cutoff frequency in Hz (default 0.5 Hz)
    
    Returns:
        Filtered ECG signal with same shape as input
    """
    # Create 4th-order Butterworth high-pass filter
    sos = signal.butter(4, cutoff, btype='high', fs=sampling_rate, output='sos')
    
    # Handle both single signal and batch
    if ecg_signal.ndim == 2:
        # Single signal: (timesteps, leads)
        filtered = signal.sosfilt(sos, ecg_signal, axis=0)
    elif ecg_signal.ndim == 3:
        # Batch: (n_samples, timesteps, leads)
        filtered = np.array([signal.sosfilt(sos, sig, axis=0) for sig in ecg_signal])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {ecg_signal.shape}")
    
    return filtered


def standardize_signals(X_train, X_val, X_test, scaler_path=None):
    """
    Standardize ECG signals using lead-wise Z-score normalization
    Fits scaler on training data only, applies to all sets
    
    Args:
        X_train: Training signals (n_train, timesteps, leads)
        X_val: Validation signals (n_val, timesteps, leads)
        X_test: Test signals (n_test, timesteps, leads)
        scaler_path: Path to save the scaler (optional)
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    n_train, n_timesteps, n_leads = X_train.shape
    
    # Reshape: (n_samples × timesteps, leads)
    X_train_reshaped = X_train.reshape(-1, n_leads)
    X_val_reshaped = X_val.reshape(-1, n_leads)
    X_test_reshaped = X_test.reshape(-1, n_leads)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    
    # Apply to validation and test
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Save scaler if path provided
    if scaler_path:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def verify_data_quality(X, y, split_name=""):
    """
    Verify data quality: check for NaN, Inf, and print statistics
    
    Args:
        X: ECG signals array
        y: Labels array
        split_name: Name of the split (for logging)
    """
    print(f"\n{'='*50}")
    print(f"Data Quality Check - {split_name}")
    print(f"{'='*50}")
    
    # Check for NaN and Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    print(f"Shape: {X.shape}")
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    print(f"Mean: {X.mean():.6f}")
    print(f"Std: {X.std():.6f}")
    print(f"Min: {X.min():.6f}")
    print(f"Max: {X.max():.6f}")
    
    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(y)*100:.2f}%)")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️ WARNING: Data contains NaN or Inf values!")
        return False
    
    print("✅ Data quality check passed!")
    return True


def compute_class_weights(y_train):
    """
    Compute class weights for handling class imbalance
    
    Args:
        y_train: Training labels
    
    Returns:
        Dictionary of class weights {0: weight_normal, 1: weight_mi}
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    class_weights = {int(c): w for c, w in zip(classes, weights)}
    
    print(f"\nClass weights (for loss function):")
    print(f"  Class 0 (Normal): {class_weights[0]:.3f}")
    print(f"  Class 1 (MI): {class_weights[1]:.3f}")
    
    return class_weights


def save_preprocessing_info(save_dir, train_csv, val_csv, test_csv, 
                            n_train, n_val, n_test, class_weights):
    """
    Save preprocessing metadata for reference
    
    Args:
        save_dir: Directory to save info file
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        n_train, n_val, n_test: Number of samples in each split
        class_weights: Class weights dictionary
    """
    info = {
        'train_csv': train_csv,
        'val_csv': val_csv,
        'test_csv': test_csv,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'class_weights': class_weights,
        'preprocessing_steps': [
            '1. Load ECG signals from PTB-XL (100 Hz)',
            '2. Remove baseline wander (0.5 Hz high-pass filter)',
            '3. Lead-wise Z-score normalization (fit on train only)',
            '4. Save as .npy arrays'
        ]
    }
    
    info_path = f"{save_dir}/preprocessing_info.txt"
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nPreprocessing info saved to {info_path}")