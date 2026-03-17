#!/usr/bin/env python3
"""
============================================================================
Explainability Analysis: Integrated Gradients for CNN-LSTM
SUBCLASS-AWARE: AMI vs IMI Lead Importance Comparison
ECGradCAM-Style Heatmap Visualization (matches Mamba XAI output)
============================================================================

Location:
  cnn_lstm_datasetA_certain_mi/scripts/7_xai_subclass_ami_vs_imi.py

Author : Chamath (E20342) — FYP ECG-IHD Detection
"""

import sys
sys.path.append('../../shared_utils')

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from captum.attr import IntegratedGradients
from scipy.ndimage import gaussian_filter1d

from model import CNNLSTM

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DATA_DIR = '../data/'
MODEL_PATH = '../saved_models/best_model.pth'
SUBCLASS_CSV = '../data/ptbxl_dataset_test_certain_vs_normal_with_mi_subclass.csv'
RESULTS_DIR = '../results/xai_subclass/'

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

IG_STEPS = 50
N_SAMPLES_PER_SUBCLASS = 5
SMOOTH_SIGMA = 12

JET = plt.cm.jet

os.makedirs(RESULTS_DIR + 'figures/', exist_ok=True)
os.makedirs(RESULTS_DIR + 'metrics/', exist_ok=True)

print(f"\n{'='*70}")
print("SUBCLASS-AWARE XAI — AMI vs IMI")
print("Model: CNN-LSTM | Dataset A (Certain MI)")
print(f"{'='*70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL WRAPPER FOR CAPTUM                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ModelWrapper(nn.Module):
    """Wrapper for CNN-LSTM: enables LSTM backward pass while freezing dropout/BN."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d)):
                module.eval()
        output = self.model(x)
        self.model.eval()
        return output


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD MODEL, DATA, AND SUBCLASS LABELS                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 1] Loading model, data, and subclass labels...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# CNN-LSTM loads from checkpoint dict
model = CNNLSTM()
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"  ✅ Model loaded: {MODEL_PATH}")

# Load test data — CNN-LSTM uses (N, 1000, 12) directly
X_test = np.load(DATA_DIR + 'X_test.npy')        # (N, 1000, 12)
y_test = np.load(DATA_DIR + 'y_test.npy')

# Load subclass CSV
df_sub = pd.read_csv(SUBCLASS_CSV)
test_ecg_ids = df_sub['ecg_id'].values
ecg_id_to_idx = {eid: i for i, eid in enumerate(test_ecg_ids)}

# Get model predictions — CNN-LSTM outputs sigmoid probabilities directly
X_torch = torch.FloatTensor(X_test).to(device)    # (N, 1000, 12) — no transpose
with torch.no_grad():
    y_proba = model(X_torch).cpu().numpy()
y_pred = (y_proba >= 0.5).astype(int)

print(f"  ✅ Test data: {X_test.shape}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: SELECT PURE AMI AND IMI SAMPLES                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 2] Selecting pure AMI and IMI samples...")


def select_subclass_samples(subclass_name, n=N_SAMPLES_PER_SUBCLASS):
    """Select top-n highest confidence correctly classified samples for a subclass."""
    mask = df_sub['mi_diagnostic_subclass'] == subclass_name
    sub_ecg_ids = df_sub.loc[mask, 'ecg_id'].values

    candidates = []
    for eid in sub_ecg_ids:
        if eid not in ecg_id_to_idx:
            continue
        idx = ecg_id_to_idx[eid]
        if y_test[idx] == 1 and y_pred[idx] == 1:
            candidates.append((idx, y_proba[idx], eid))

    # Filter: avoid saturated sigmoid (100%) — gradients are near-zero
    # Keep 85-99% range where IG attributions are reliable
    candidates = [(idx, prob, eid) for idx, prob, eid in candidates
                  if 0.85 <= prob <= 0.99]
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[:n]

    print(f"  {subclass_name}: {len(candidates)} correct, selected top {len(selected)}")
    for idx, prob, eid in selected:
        print(f"    ecg_id={eid:>6d}  prob={prob:.1%}")

    return selected


ami_samples = select_subclass_samples('AMI')
imi_samples = select_subclass_samples('IMI')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: COMPUTE ATTRIBUTIONS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 3] Computing Integrated Gradients...")

model_wrapper = ModelWrapper(model)
ig = IntegratedGradients(model_wrapper)


def compute_attributions(ecg_sample):
    """Compute IG for a single ECG. Input/output: (1000, 12)."""
    # CNN-LSTM expects (B, 1000, 12) — same as numpy shape
    inp = torch.FloatTensor(ecg_sample).unsqueeze(0).to(device)
    inp.requires_grad = True
    baseline = torch.zeros_like(inp).to(device)
    attrs = ig.attribute(inp, baselines=baseline, n_steps=IG_STEPS)
    return attrs.squeeze(0).cpu().detach().numpy()  # (1000, 12)


def smooth_attributions(attrs, sigma=SMOOTH_SIGMA):
    smoothed = np.zeros_like(attrs)
    for lead in range(attrs.shape[1]):
        smoothed[:, lead] = gaussian_filter1d(np.abs(attrs[:, lead]), sigma=sigma)
    return smoothed


# Compute for AMI
print(f"  Computing AMI attributions ({len(ami_samples)} samples)...")
ami_attrs = []
for i, (idx, prob, eid) in enumerate(ami_samples):
    print(f"    [{i+1}/{len(ami_samples)}] ecg_id={eid}")
    ami_attrs.append(compute_attributions(X_test[idx]))

# Compute for IMI
print(f"  Computing IMI attributions ({len(imi_samples)} samples)...")
imi_attrs = []
for i, (idx, prob, eid) in enumerate(imi_samples):
    print(f"    [{i+1}/{len(imi_samples)}] ecg_id={eid}")
    imi_attrs.append(compute_attributions(X_test[idx]))

print(f"  ✅ All attributions computed")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: LEAD IMPORTANCE COMPARISON (AMI vs IMI)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 4] Computing lead importance per subclass...")


def compute_lead_importance(attrs_list):
    importances = []
    for attrs in attrs_list:
        lead_imp = np.abs(attrs).sum(axis=0)
        lead_imp_norm = lead_imp / lead_imp.sum()
        importances.append(lead_imp_norm)
    importances = np.array(importances)
    return importances.mean(axis=0), importances.std(axis=0)


ami_avg, ami_std = compute_lead_importance(ami_attrs)
imi_avg, imi_std = compute_lead_importance(imi_attrs)

# Save CSV
comp_df = pd.DataFrame({
    'Lead': LEAD_NAMES,
    'AMI_Importance': ami_avg, 'AMI_Std': ami_std,
    'IMI_Importance': imi_avg, 'IMI_Std': imi_std,
})
comp_df.to_csv(RESULTS_DIR + 'metrics/lead_importance_ami_vs_imi.csv', index=False)
print(f"  ✅ Saved: lead_importance_ami_vs_imi.csv")

for name, avg in [('AMI', ami_avg), ('IMI', imi_avg)]:
    ranked = np.argsort(avg)[::-1]
    top3 = [LEAD_NAMES[i] for i in ranked[:3]]
    print(f"  {name} top 3: {', '.join(top3)}")


# ── Side-by-side bar chart ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
x = np.arange(12)
bar_width = 0.65

# AMI
ami_sorted = np.argsort(ami_avg)[::-1]
ami_colors = ['#2171b5'] * 12
for rank, idx_val in enumerate(ami_sorted[:3]):
    ami_colors[idx_val] = ['#c0392b', '#e67e22', '#f1c40f'][rank]

axes[0].bar(x, ami_avg, yerr=ami_std, color=ami_colors, alpha=0.88,
            edgecolor='#2c3e50', linewidth=1.2, capsize=4, width=bar_width)
axes[0].set_xticks(x)
axes[0].set_xticklabels(LEAD_NAMES, fontsize=10, fontweight='bold')
axes[0].set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
axes[0].set_title(f'AMI (Anterior MI)\nTop: {", ".join([LEAD_NAMES[i] for i in ami_sorted[:3]])}',
                   fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, max(ami_avg.max(), imi_avg.max()) * 1.35)

for i, v in enumerate(ami_avg):
    axes[0].text(i, v + ami_std[i] + 0.003, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

# IMI
imi_sorted = np.argsort(imi_avg)[::-1]
imi_colors = ['#2171b5'] * 12
for rank, idx_val in enumerate(imi_sorted[:3]):
    imi_colors[idx_val] = ['#c0392b', '#e67e22', '#f1c40f'][rank]

axes[1].bar(x, imi_avg, yerr=imi_std, color=imi_colors, alpha=0.88,
            edgecolor='#2c3e50', linewidth=1.2, capsize=4, width=bar_width)
axes[1].set_xticks(x)
axes[1].set_xticklabels(LEAD_NAMES, fontsize=10, fontweight='bold')
axes[1].set_title(f'IMI (Inferior MI)\nTop: {", ".join([LEAD_NAMES[i] for i in imi_sorted[:3]])}',
                   fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('CNN-LSTM: Lead Importance by MI Subclass\n'
             f'(Dataset A — {N_SAMPLES_PER_SUBCLASS} samples per subclass)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/lead_importance_ami_vs_imi.png',
            dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: lead_importance_ami_vs_imi.png")
plt.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: REPRESENTATIVE HEATMAPS (ECGradCAM style)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 5] Creating representative heatmaps...")


def plot_ecg_heatmap_panel(ax, ecg_signal, attr_smooth, lead_name,
                           global_vmax=None, show_xlabel=False):
    """ECGradCAM-style heatmap panel — matches Mamba XAI output exactly."""
    L = len(ecg_signal)
    vmax = global_vmax if global_vmax is not None else (attr_smooth.max() + 1e-8)
    a_norm = np.clip(attr_smooth / vmax, 0, 1)
    heatmap_2d = np.tile(a_norm, (100, 1))

    ptp = ecg_signal.max() - ecg_signal.min()
    y_lo = ecg_signal.min() - 0.15 * (ptp + 1e-8)
    y_hi = ecg_signal.max() + 0.15 * (ptp + 1e-8)

    ax.imshow(heatmap_2d, aspect='auto', cmap=JET, extent=[0, L, y_lo, y_hi],
              origin='lower', alpha=0.88, interpolation='bilinear', vmin=0, vmax=1)
    ax.plot(np.arange(L), ecg_signal, color='black', linewidth=1.3, zorder=10)
    ax.set_xlim(0, L); ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel(lead_name, fontsize=9, fontweight='bold', rotation=0, labelpad=25, va='center')
    ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    for s in ax.spines.values(): s.set_visible(False)
    if show_xlabel: ax.set_xlabel('Time (samples @ 100 Hz)', fontsize=9)


# --- Best AMI and IMI samples ---
best_ami_idx, best_ami_prob, best_ami_eid = ami_samples[0]
best_ami_attrs = smooth_attributions(ami_attrs[0])

best_imi_idx, best_imi_prob, best_imi_eid = imi_samples[0]
best_imi_attrs = smooth_attributions(imi_attrs[0])

global_vmax = max(best_ami_attrs.max(), best_imi_attrs.max())

# --- Side-by-side 12-lead: AMI vs IMI ---
fig, axes = plt.subplots(12, 2, figsize=(20, 20), facecolor='white')
fig.suptitle(
    f'CNN-LSTM Integrated Gradients — AMI vs IMI Comparison\n'
    f'AMI (ecg {best_ami_eid}, {best_ami_prob:.1%})                    '
    f'IMI (ecg {best_imi_eid}, {best_imi_prob:.1%})',
    fontsize=15, fontweight='bold', y=0.995
)

for i in range(12):
    plot_ecg_heatmap_panel(
        axes[i, 0], X_test[best_ami_idx][:, i], best_ami_attrs[:, i],
        LEAD_NAMES[i], global_vmax=global_vmax, show_xlabel=(i == 11))
    plot_ecg_heatmap_panel(
        axes[i, 1], X_test[best_imi_idx][:, i], best_imi_attrs[:, i],
        LEAD_NAMES[i], global_vmax=global_vmax, show_xlabel=(i == 11))

axes[0, 0].set_title('AMI (Anterior MI)', fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_title('IMI (Inferior MI)', fontsize=13, fontweight='bold', pad=10)

sm = plt.cm.ScalarMappable(cmap=JET, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.12, 0.012, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Model Attention', fontsize=11, fontweight='bold')
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(['Low', 'Med', 'High'])

plt.tight_layout(rect=[0.03, 0, 0.92, 0.97])
plt.savefig(RESULTS_DIR + 'figures/heatmap_ami_vs_imi_12lead.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"  ✅ Saved: heatmap_ami_vs_imi_12lead.png")
plt.close()


# --- Clinically relevant leads grid (2×4) ---
AMI_LEADS = [6, 7, 8, 9]    # V1, V2, V3, V4
IMI_LEADS = [1, 2, 5, 11]   # II, III, aVF, V6

fig, axes = plt.subplots(2, 4, figsize=(22, 8), facecolor='white')
fig.suptitle(
    f'CNN-LSTM: Clinically Relevant Leads — AMI vs IMI\n'
    f'AMI (ecg {best_ami_eid}, {best_ami_prob:.1%})  |  '
    f'IMI (ecg {best_imi_eid}, {best_imi_prob:.1%})',
    fontsize=14, fontweight='bold'
)

for i, lead_idx in enumerate(AMI_LEADS):
    plot_ecg_heatmap_panel(
        axes[0, i], X_test[best_ami_idx][:, lead_idx], best_ami_attrs[:, lead_idx],
        f'Lead {LEAD_NAMES[lead_idx]}', global_vmax=global_vmax, show_xlabel=False)
    axes[0, i].set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')

for i, lead_idx in enumerate(IMI_LEADS):
    plot_ecg_heatmap_panel(
        axes[1, i], X_test[best_imi_idx][:, lead_idx], best_imi_attrs[:, lead_idx],
        f'Lead {LEAD_NAMES[lead_idx]}', global_vmax=global_vmax, show_xlabel=True)
    axes[1, i].set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')

axes[0, 0].annotate('AMI', xy=(-0.35, 0.5), xycoords='axes fraction',
                     fontsize=14, fontweight='bold', color='#c0392b',
                     ha='center', va='center', rotation=90)
axes[1, 0].annotate('IMI', xy=(-0.35, 0.5), xycoords='axes fraction',
                     fontsize=14, fontweight='bold', color='#2980b9',
                     ha='center', va='center', rotation=90)

sm = plt.cm.ScalarMappable(cmap=JET, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Importance', fontsize=11, fontweight='bold')
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(['Low', 'Med', 'High'])

plt.tight_layout(rect=[0.04, 0, 0.93, 0.94])
plt.savefig(RESULTS_DIR + 'figures/heatmap_ami_vs_imi_relevant_leads.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"  ✅ Saved: heatmap_ami_vs_imi_relevant_leads.png")
plt.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ami_top3 = [LEAD_NAMES[i] for i in np.argsort(ami_avg)[::-1][:3]]
imi_top3 = [LEAD_NAMES[i] for i in np.argsort(imi_avg)[::-1][:3]]

print(f"\n{'='*70}")
print("SUBCLASS XAI COMPLETE — CNN-LSTM")
print(f"{'='*70}")
print(f"""
📊 Lead Importance:
   AMI top 3: {', '.join(ami_top3)}   (expected: V1, V2, V3)
   IMI top 3: {', '.join(imi_top3)}   (expected: II, III, aVF)

📁 Outputs:
   figures/
     • lead_importance_ami_vs_imi.png
     • heatmap_ami_vs_imi_12lead.png
     • heatmap_ami_vs_imi_relevant_leads.png
   metrics/
     • lead_importance_ami_vs_imi.csv

💡 Compare with Mamba-2 results for cross-architecture analysis.
{'='*70}
""")