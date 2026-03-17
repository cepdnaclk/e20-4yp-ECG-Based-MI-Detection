"""
Evaluation Script - Certain MI Only Test Set
Evaluates trained model on restricted test set: pure_normal vs certain_mi only
(removes uncertain_mi to address evaluation bias)
"""

import sys
sys.path.append('../../shared_utils')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNNLSTM

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = '../data/'
MODEL_PATH = '../saved_models/best_model.pth'
RESULTS_DIR = '../results/certain_only/'

# Create results directory
import os
os.makedirs(RESULTS_DIR + 'figures/', exist_ok=True)
os.makedirs(RESULTS_DIR + 'metrics/', exist_ok=True)

print(f"\n{'='*60}")
print("EVALUATION - CERTAIN MI ONLY TEST SET")
print(f"{'='*60}")

# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================

print(f"\n[Step 1] Loading preprocessed test data...")

# Load full test data
X_test_full = np.load(DATA_DIR + 'X_test.npy')
y_test_full = np.load(DATA_DIR + 'y_test.npy')
groups_test_full = np.load(DATA_DIR + 'groups_test.npy', allow_pickle=True)

print(f"✅ Full test data loaded: {X_test_full.shape}")

# ==========================================
# 2. FILTER TO CERTAIN MI ONLY
# ==========================================

print(f"\n[Step 2] Filtering to certain MI only...")

# Create mask: keep only pure_normal and certain_mi
mask = (groups_test_full == 'pure_normal') | (groups_test_full == 'certain_mi')

X_test = X_test_full[mask]
y_test = y_test_full[mask]
groups_test = groups_test_full[mask]

print(f"✅ Filtered test set: {X_test.shape}")
print(f"\nGroup distribution:")
unique, counts = np.unique(groups_test, return_counts=True)
for group, count in zip(unique, counts):
    print(f"  {group}: {count}")

# Verify no uncertain_mi
assert 'uncertain_mi' not in groups_test, "Error: uncertain_mi found in filtered data!"
print(f"\n✅ Verified: No uncertain_mi in test set")

# ==========================================
# 3. LOAD MODEL
# ==========================================

print(f"\n[Step 3] Loading trained model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model (CNNLSTM takes no arguments)
model = CNNLSTM()

# Load checkpoint (contains model_state_dict, optimizer, etc.)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Extract just the model weights
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
else:
    # If it's just the state dict directly
    model.load_state_dict(checkpoint)
    
model.to(device)
model.eval()

print(f"✅ Model loaded from: {MODEL_PATH}")

# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================

print(f"\n[Step 4] Generating predictions...")

X_test_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    y_pred_proba = model(X_test_tensor).cpu().numpy()

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"✅ Predictions generated")
print(f"  Predicted Positive (MI): {y_pred.sum()}")
print(f"  Predicted Negative (Normal): {(y_pred == 0).sum()}")

# ==========================================
# 5. COMPUTE METRICS
# ==========================================

print(f"\n{'='*60}")
print("OVERALL PERFORMANCE - CERTAIN MI ONLY")
print(f"{'='*60}")

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity, PPV, NPV
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n📊 Classification Metrics:")
print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
print(f"  ROC-AUC:     {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  PPV:         {ppv:.4f} ({ppv*100:.2f}%)")
print(f"  NPV:         {npv:.4f} ({npv*100:.2f}%)")

print(f"\n📊 Confusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

# ==========================================
# 6. GROUP-WISE ANALYSIS
# ==========================================

print(f"\n{'='*60}")
print("GROUP-WISE PERFORMANCE")
print(f"{'='*60}")

# Certain MI vs Pure Normal
certain_mi_mask = groups_test == 'certain_mi'
pure_normal_mask = groups_test == 'pure_normal'

# Certain MI metrics
certain_mi_recall = recall_score(y_test[certain_mi_mask], y_pred[certain_mi_mask])
certain_mi_n = certain_mi_mask.sum()
certain_mi_tp = ((y_test[certain_mi_mask] == 1) & (y_pred[certain_mi_mask] == 1)).sum()
certain_mi_fn = ((y_test[certain_mi_mask] == 1) & (y_pred[certain_mi_mask] == 0)).sum()

print(f"\nCertain MI (n={certain_mi_n}):")
print(f"  Recall:          {certain_mi_recall:.4f} ({certain_mi_recall*100:.2f}%)")
print(f"  Detected (TP):   {certain_mi_tp}")
print(f"  Missed (FN):     {certain_mi_fn}")

# Pure Normal metrics
normal_specificity = (y_test[pure_normal_mask] == y_pred[pure_normal_mask]).sum() / pure_normal_mask.sum()
normal_n = pure_normal_mask.sum()
normal_tn = ((y_test[pure_normal_mask] == 0) & (y_pred[pure_normal_mask] == 0)).sum()
normal_fp = ((y_test[pure_normal_mask] == 0) & (y_pred[pure_normal_mask] == 1)).sum()

print(f"\nPure Normal (n={normal_n}):")
print(f"  Specificity:     {normal_specificity:.4f} ({normal_specificity*100:.2f}%)")
print(f"  Correct (TN):    {normal_tn}")
print(f"  False Alarm (FP): {normal_fp}")

# ==========================================
# 7. SAVE RESULTS
# ==========================================

print(f"\n[Step 5] Saving results...")

# Save predictions
results_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'y_prob': y_pred_proba,
    'group': groups_test
})
results_df.to_csv(RESULTS_DIR + 'metrics/test_predictions_certain_only.csv', index=False)
print(f"✅ Saved: test_predictions_certain_only.csv")

# Save metrics
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
               'specificity', 'ppv', 'npv'],
    'value': [accuracy, precision, recall, f1, roc_auc, 
              specificity, ppv, npv]
})
metrics_df.to_csv(RESULTS_DIR + 'metrics/test_metrics_certain_only.csv', index=False)
print(f"✅ Saved: test_metrics_certain_only.csv")

# Save group metrics
group_metrics_df = pd.DataFrame({
    'group': ['certain_mi', 'pure_normal'],
    'n': [certain_mi_n, normal_n],
    'recall_or_specificity': [certain_mi_recall, normal_specificity],
    'tp_or_tn': [certain_mi_tp, normal_tn],
    'fn_or_fp': [certain_mi_fn, normal_fp]
})
group_metrics_df.to_csv(RESULTS_DIR + 'metrics/test_group_metrics_certain_only.csv', index=False)
print(f"✅ Saved: test_group_metrics_certain_only.csv")

# ==========================================
# 8. GENERATE PLOTS
# ==========================================

print(f"\n[Step 6] Generating plots...")

# Plot 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'MI'], yticklabels=['Normal', 'MI'])
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Certain MI Only Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/confusion_matrix_certain_only.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: confusion_matrix_certain_only.png")
plt.close()

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Certain MI Only Test Set', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/roc_curve_certain_only.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: roc_curve_certain_only.png")
plt.close()

# Plot 3: Probability Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Normal', color='green', edgecolor='black')
ax.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Certain MI', color='red', edgecolor='black')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prediction Distribution - Certain MI Only Test Set', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/probability_distribution_certain_only.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: probability_distribution_certain_only.png")
plt.close()

print(f"\n{'='*60}")
print("EVALUATION COMPLETE - CERTAIN MI ONLY")
print(f"{'='*60}")
print(f"\nResults saved in: {RESULTS_DIR}")
print(f"  • Metrics CSVs: {RESULTS_DIR}metrics/")
print(f"  • Figures: {RESULTS_DIR}figures/")
print(f"\n{'='*60}\n")