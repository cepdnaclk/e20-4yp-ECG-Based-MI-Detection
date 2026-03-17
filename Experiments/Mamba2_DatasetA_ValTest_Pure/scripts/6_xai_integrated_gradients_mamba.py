#!/usr/bin/env python3
"""
============================================================================
Explainability Analysis: Integrated Gradients for Bidirectional Mamba-2
ECG MI Detection — ECGradCAM-Style Heatmap Visualization
============================================================================

Generates publication-quality attention maps matching the visual style from:
  Hicks et al. (2021) "Explaining deep neural networks for knowledge
  discovery in electrocardiogram analysis" — Nature Scientific Reports

Visual style: JET colormap (deep blue → cyan → green → yellow → red)
              with ECG signal overlaid as a dark line.

Outputs:
  1. 12-lead heatmap overlays per sample
  2. Selected 4-lead panels (Lead II, aVF, V2, V5) in 2×2 grid
  3. Lead-wise importance bar chart (averaged over MI cases)
  4. Temporal importance profile
  5. CSV metrics for cross-model comparison

Target location:
  /scratch1/e20-fyp-ecg-ihd-detection/experiments_new/
      Mamba2_DatasetA_ValTest_Pure/scripts/6_xai_integrated_gradients_mamba.py

Author : Chamath (E20342) — FYP ECG-IHD Detection
Date   : 2025
"""

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from captum.attr import IntegratedGradients
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ---  Paths (relative to scripts/ directory) ---
DATA_DIR = '../../cnn_lstm_datasetA_certain_mi/data/'   # shared preprocessed data
MODEL_PATH = '../models/datasetA_best.pt'               # Mamba best weights
RESULTS_DIR = '../results/xai/'

# --- ECG lead names (standard 12-lead order) ---
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Clinically important leads for MI ---
SELECTED_LEADS = [1, 5, 7, 10]  # Lead II, aVF, V2, V5

# --- IG settings ---
IG_STEPS = 50           # integration steps
N_MI_FOR_LEAD_AVG = 30  # MI cases for lead importance averaging

# Create output dirs
os.makedirs(RESULTS_DIR + 'figures/', exist_ok=True)
os.makedirs(RESULTS_DIR + 'metrics/', exist_ok=True)

print(f"\n{'='*70}")
print("EXPLAINABILITY ANALYSIS — INTEGRATED GRADIENTS")
print("Model: Bidirectional Mamba-2 (Selective SSM)")
print("Style: ECGradCAM (Hicks et al., 2021)")
print(f"{'='*70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL DEFINITION  (self-contained, no external model.py needed)       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class CFG:
    """Exact config used during Mamba-2 training on Dataset A."""
    D_MODEL: int = 256
    D_STATE: int = 64
    D_CONV: int = 4
    EXPAND: int = 2
    N_LAYERS: int = 6
    DROPOUT: float = 0.2
    POOLING: str = "attn"
    MODEL_TYPE: str = "mamba2"
    BIDIRECTIONAL: bool = True

cfg = CFG()

# --- Import official Mamba-2 ---
try:
    from mamba_ssm import Mamba, Mamba2
    _HAVE_MAMBA = True
    print(f"  ✅ mamba_ssm available")
except ImportError:
    _HAVE_MAMBA = False
    print(f"  ❌ mamba_ssm not found — install with: pip install mamba-ssm")
    sys.exit(1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight.view(1, 1, -1)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = expansion * d_model
        self.fc1 = nn.Linear(d_model, 2 * hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        u, v = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(u) * v
        x = self.fc2(x)
        return self.drop(x)


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        w = self.score(x).squeeze(-1)
        w = torch.softmax(w, dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class MambaMixer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, model_type="mamba"):
        super().__init__()
        if model_type == "mamba":
            self.mixer = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        elif model_type == "mamba2":
            self.mixer = Mamba2(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        return self.mixer(x)


class BiMixer(nn.Module):
    def __init__(self, mixer_fwd, mixer_bwd=None, d_model=256, mode="concat"):
        super().__init__()
        self.fwd = mixer_fwd
        self.bwd = mixer_bwd if mixer_bwd is not None else mixer_fwd
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            self.proj = None

    def forward(self, x):
        y_f = self.fwd(x)
        xr = torch.flip(x, dims=[1])
        y_b = self.bwd(xr)
        y_b = torch.flip(y_b, dims=[1])

        if self.mode == "avg":
            return 0.5 * (y_f + y_b)
        elif self.mode == "sum":
            return y_f + y_b
        elif self.mode == "concat":
            return self.proj(torch.cat([y_f, y_b], dim=-1))
        else:
            raise ValueError(self.mode)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout,
                 model_type, bidirectional, mlp_expand=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        mixer = MambaMixer(d_model, d_state, d_conv, expand, model_type)

        if bidirectional:
            mixer_b = MambaMixer(d_model, d_state, d_conv, expand, model_type)
            self.mixer = BiMixer(mixer, mixer_b, d_model=d_model, mode="concat")
        else:
            self.mixer = mixer

        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, expansion=mlp_expand, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.mixer(self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class ECGMambaClassifier(nn.Module):
    """Bidirectional Mamba-2 classifier for 12-lead ECG binary classification."""
    def __init__(self, in_ch: int = 12):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, cfg.D_MODEL)
        self.drop_in = nn.Dropout(cfg.DROPOUT)

        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=cfg.D_MODEL, d_state=cfg.D_STATE,
                d_conv=cfg.D_CONV, expand=cfg.EXPAND,
                dropout=cfg.DROPOUT, model_type=cfg.MODEL_TYPE,
                bidirectional=cfg.BIDIRECTIONAL, mlp_expand=4
            )
            for _ in range(cfg.N_LAYERS)
        ])

        self.norm_out = RMSNorm(cfg.D_MODEL)
        self.pool = AttnPool(cfg.D_MODEL) if cfg.POOLING == "attn" else None
        self.head = nn.Linear(cfg.D_MODEL, 1)

    def forward(self, x):
        # x: (B, 12, L) from dataloader
        x = x.transpose(1, 2)                  # → (B, L, 12)
        x = self.drop_in(self.in_proj(x))      # → (B, L, D_MODEL)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_out(x)
        feat = x.mean(dim=1) if self.pool is None else self.pool(x)
        return self.head(feat).squeeze(-1)      # raw logits


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL WRAPPER FOR CAPTUM                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MambaIGWrapper(nn.Module):
    """
    Wrapper for Captum's IntegratedGradients.

    Handles two Mamba-specific issues:
      1) Output is raw logits → apply sigmoid so IG attributes toward P(MI)
      2) Mamba2 CUDA kernels need train mode for backward pass,
         but we freeze dropout/batchnorm-like layers in eval mode
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Enable train mode for Mamba2 backward compatibility
        self.model.train()

        # Keep dropout and norm layers in eval mode
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d, RMSNorm)):
                module.eval()

        logits = self.model(x)
        self.model.eval()

        return torch.sigmoid(logits)  # return P(MI)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD MODEL AND DATA                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 1] Loading Mamba-2 model and test data...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# Build model and load weights
model = ECGMambaClassifier().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.eval()
print(f"  ✅ Model loaded from: {MODEL_PATH}")
print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load preprocessed test data
X_test = np.load(DATA_DIR + 'X_test.npy')        # (N, 1000, 12)
y_test = np.load(DATA_DIR + 'y_test.npy')
groups_test = np.load(DATA_DIR + 'groups_test.npy', allow_pickle=True)

print(f"  ✅ Test data: {X_test.shape}  |  MI: {int(y_test.sum())}  Normal: {int((y_test==0).sum())}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: SELECT REPRESENTATIVE SAMPLES                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 2] Selecting representative ECG samples...")

# Get model predictions (Mamba outputs logits → need sigmoid)
X_test_torch = torch.FloatTensor(X_test.transpose(0, 2, 1)).to(device)  # (N,12,1000)

with torch.no_grad():
    logits = model(X_test_torch).cpu().numpy()
    y_pred_proba = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

y_pred = (y_pred_proba >= 0.5).astype(int)

# Find representative examples
mi_correct_idx   = np.where((y_test == 1) & (y_pred == 1) & (groups_test == 'certain_mi'))[0]
mi_missed_idx    = np.where((y_test == 1) & (y_pred == 0))[0]
normal_correct_idx = np.where((y_test == 0) & (y_pred == 0))[0]
false_positive_idx = np.where((y_test == 0) & (y_pred == 1))[0]

examples = {
    'MI_Correct_HighConf':  mi_correct_idx[np.argmax(y_pred_proba[mi_correct_idx])] if len(mi_correct_idx) > 0 else None,
    'MI_Correct_MedConf':   mi_correct_idx[len(mi_correct_idx)//2] if len(mi_correct_idx) > 0 else None,
    'MI_Missed':            mi_missed_idx[0] if len(mi_missed_idx) > 0 else None,
    'Normal_Correct':       normal_correct_idx[0] if len(normal_correct_idx) > 0 else None,
    'False_Positive':       false_positive_idx[0] if len(false_positive_idx) > 0 else None,
}

print(f"\n  {'Example':<25s} {'Pred':>6s}  {'Prob':>7s}  {'Truth':>6s}")
print(f"  {'─'*50}")
for name, idx in examples.items():
    if idx is not None:
        pred = "MI" if y_pred[idx] == 1 else "Normal"
        truth = "MI" if y_test[idx] == 1 else "Normal"
        print(f"  {name:<25s} {pred:>6s}  {y_pred_proba[idx]:>6.1%}  {truth:>6s}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: INTEGRATED GRADIENTS COMPUTATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 3] Computing Integrated Gradients attributions...")

model_wrapper = MambaIGWrapper(model)
ig = IntegratedGradients(model_wrapper)


def compute_attributions(ecg_sample):
    """
    Compute IG attributions for a single ECG sample.

    Args:
        ecg_sample: (1000, 12) numpy array

    Returns:
        attributions: (1000, 12) numpy array — importance per timestep per lead
    """
    # Model expects (B, 12, L) — transpose from (1000,12) → (12,1000)
    inp = torch.FloatTensor(ecg_sample.T).unsqueeze(0).to(device)  # (1,12,1000)
    inp.requires_grad = True

    baseline = torch.zeros_like(inp).to(device)

    attrs = ig.attribute(inp, baselines=baseline, n_steps=IG_STEPS)

    # attrs shape: (1, 12, 1000) → transpose back to (1000, 12)
    return attrs.squeeze(0).permute(1, 0).cpu().detach().numpy()


# Compute for all selected examples
attributions_dict = {}
for name, idx in examples.items():
    if idx is not None:
        print(f"    Computing: {name}...")
        attributions_dict[name] = compute_attributions(X_test[idx])

print(f"  ✅ Attributions computed for {len(attributions_dict)} examples")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: ECGRADCAM-STYLE HEATMAP VISUALIZATIONS                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 4] Creating ECGradCAM-style heatmap overlays...")

# --- JET colormap (matches Hicks et al. 2021 / cv2.COLORMAP_JET) ---
JET = plt.cm.jet

# --- Gaussian smoothing for IG attributions ---
SMOOTH_SIGMA = 12   # samples (~120ms at 100Hz) — spreads sharp IG spikes
                     # into broader attention regions like GradCAM produces


def smooth_attributions(attrs, sigma=SMOOTH_SIGMA):
    """
    Apply Gaussian smoothing to IG attributions.

    Why: Integrated Gradients produces pixel-sharp spikes (mathematically
    precise but visually hard to read). GradCAM is naturally smooth because
    it operates on downsampled feature maps. To match the Hicks et al. style,
    we smooth IG attributions with a Gaussian kernel.

    sigma=12 ≈ 120ms window — roughly one QRS width, producing attention
    blobs similar to GradCAM's resolution.
    """
    smoothed = np.zeros_like(attrs)
    for lead in range(attrs.shape[1]):
        smoothed[:, lead] = gaussian_filter1d(np.abs(attrs[:, lead]), sigma=sigma)
    return smoothed


def plot_ecg_heatmap_panel(ax, ecg_signal, attr_smooth, lead_name,
                           global_vmax=None, show_xlabel=False):
    """
    Draw a single lead panel in ECGradCAM style (Hicks et al. 2021).

    Improvements over v1:
      1) imshow-based rendering (smooth, fast, no axvspan loop)
      2) Global normalization across all 12 leads (not per-lead)
         → ensures color consistency: a red region in V2 means the same
           intensity as red in Lead I
      3) Smoothed attributions input (Gaussian pre-filtered)
      4) Black ECG line with slight thickening for visibility
    """
    L = len(ecg_signal)

    # --- Normalize using global max (cross-lead consistency) ---
    vmax = global_vmax if global_vmax is not None else (attr_smooth.max() + 1e-8)
    a_norm = attr_smooth / vmax
    a_norm = np.clip(a_norm, 0, 1)

    # --- Build 2D heatmap image for imshow ---
    # Tile the 1D attribution into a vertical band (100 pixels tall)
    heatmap_2d = np.tile(a_norm, (100, 1))  # (100, L)

    # --- Render heatmap as background ---
    ptp = ecg_signal.max() - ecg_signal.min()
    y_lo = ecg_signal.min() - 0.15 * (ptp + 1e-8)
    y_hi = ecg_signal.max() + 0.15 * (ptp + 1e-8)
    extent = [0, L, y_lo, y_hi]

    ax.imshow(heatmap_2d, aspect='auto', cmap=JET, extent=extent,
              origin='lower', alpha=0.88, interpolation='bilinear',
              vmin=0, vmax=1)

    # --- ECG signal overlay (dark line, visible on both blue and red) ---
    t = np.arange(L)
    ax.plot(t, ecg_signal, color='black', linewidth=1.3, zorder=10)

    # --- Clean axes (ECGradCAM style — no spines, no ticks) ---
    ax.set_xlim(0, L)
    ax.set_ylim(y_lo, y_hi)

    ax.set_ylabel(lead_name, fontsize=10, fontweight='bold',
                  rotation=0, labelpad=28, va='center')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_xlabel:
        ax.set_xlabel('Time (samples @ 100 Hz)', fontsize=10)


# ── 4a. Full 12-lead heatmap ──────────────────────────────────────────────

def create_12lead_heatmap(ecg, attrs, sample_name, pred_prob, true_label):
    """
    12 leads stacked vertically, ECGradCAM style.
    Uses GLOBAL normalization so all leads share the same color scale.
    """
    # Smooth attributions
    attrs_smooth = smooth_attributions(attrs)

    # Global vmax across all 12 leads (key for cross-lead comparison)
    global_vmax = attrs_smooth.max()

    fig, axes = plt.subplots(12, 1, figsize=(14, 18), facecolor='white')
    fig.suptitle(
        f'Mamba-2 Integrated Gradients — {sample_name}\n'
        f'Predicted MI Probability: {pred_prob:.1%}   |   '
        f'True Label: {"MI" if true_label == 1 else "Normal"}',
        fontsize=14, fontweight='bold', y=0.995
    )

    for i in range(12):
        plot_ecg_heatmap_panel(
            axes[i], ecg[:, i], attrs_smooth[:, i],
            lead_name=LEAD_NAMES[i],
            global_vmax=global_vmax,
            show_xlabel=(i == 11)
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=JET, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Model Attention', fontsize=11, fontweight='bold')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', 'Med', 'High'])

    plt.tight_layout(rect=[0.04, 0, 0.90, 0.97])

    fname = f"{sample_name.replace(' ', '_')}_12lead_heatmap.png"
    plt.savefig(RESULTS_DIR + 'figures/' + fname, dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"    ✅ {fname}")
    plt.close()


# ── 4b. Selected 4-lead grid (2×2, like Figure 4 in Hicks et al.) ────────

def create_4lead_grid(ecg, attrs, sample_name, pred_prob, true_label,
                      leads=SELECTED_LEADS):
    """2×2 grid of clinically important leads — publication figure style."""

    attrs_smooth = smooth_attributions(attrs)
    # Global vmax across selected leads only
    global_vmax = max(attrs_smooth[:, l].max() for l in leads)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), facecolor='white')
    fig.suptitle(
        f'Mamba-2 — {sample_name}\n'
        f'Predicted: {pred_prob:.1%} MI   |   '
        f'True: {"MI" if true_label == 1 else "Normal"}',
        fontsize=13, fontweight='bold'
    )

    for i, lead_idx in enumerate(leads):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        plot_ecg_heatmap_panel(
            ax, ecg[:, lead_idx], attrs_smooth[:, lead_idx],
            lead_name=f'Lead {LEAD_NAMES[lead_idx]}',
            global_vmax=global_vmax,
            show_xlabel=(row == 1)
        )
        ax.set_title(f'Lead {LEAD_NAMES[lead_idx]}', fontsize=11,
                     fontweight='bold', pad=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=JET, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Importance', fontsize=11, fontweight='bold')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', 'Med', 'High'])

    plt.tight_layout(rect=[0.02, 0, 0.91, 0.94])

    fname = f"{sample_name.replace(' ', '_')}_4lead_grid.png"
    plt.savefig(RESULTS_DIR + 'figures/' + fname, dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"    ✅ {fname}")
    plt.close()


# Generate all visualizations for each example
for name, idx in examples.items():
    if idx is not None and name in attributions_dict:
        ecg   = X_test[idx]
        attrs = attributions_dict[name]
        prob  = y_pred_proba[idx]
        label = y_test[idx]

        create_12lead_heatmap(ecg, attrs, name, prob, label)
        create_4lead_grid(ecg, attrs, name, prob, label)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: LEAD-WISE IMPORTANCE ANALYSIS                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 5] Computing lead-wise importance (n={N_MI_FOR_LEAD_AVG} MI cases)...")

mi_indices = np.where((y_test == 1) & (y_pred == 1))[0][:N_MI_FOR_LEAD_AVG]
lead_importances = []

for i, idx in enumerate(mi_indices):
    if (i + 1) % 10 == 0:
        print(f"    Processing {i+1}/{len(mi_indices)}...")
    attrs = compute_attributions(X_test[idx])
    lead_imp = np.abs(attrs).sum(axis=0)     # sum |attr| over time per lead
    lead_importances.append(lead_imp)

lead_importances = np.array(lead_importances)         # (N, 12)
avg_imp  = lead_importances.mean(axis=0)
std_imp  = lead_importances.std(axis=0)
avg_norm = avg_imp / avg_imp.sum()
std_norm = std_imp / avg_imp.sum()

# --- Bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(12)

# Color bars: highlight top-3 leads
sorted_idx = np.argsort(avg_norm)[::-1]
colors = ['#2171b5'] * 12
for rank, idx_val in enumerate(sorted_idx[:3]):
    colors[idx_val] = ['#c0392b', '#e67e22', '#f1c40f'][rank]

bars = ax.bar(x, avg_norm, yerr=std_norm, color=colors, alpha=0.88,
              edgecolor='#2c3e50', linewidth=1.2, capsize=5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(LEAD_NAMES, fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
ax.set_xlabel('ECG Lead', fontsize=12, fontweight='bold')
ax.set_title(f'Mamba-2: Lead-Wise Importance for MI Detection\n'
             f'(Averaged over {len(mi_indices)} correctly classified MI cases)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', zorder=0)
ax.set_ylim(0, max(avg_norm) * 1.25)
ax.set_axisbelow(True)

# Value labels on bars
for i, (imp, std) in enumerate(zip(avg_norm, std_norm)):
    ax.text(i, imp + std + 0.004, f'{imp:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

# Legend for top-3
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#c0392b', edgecolor='#2c3e50', label=f'1st: {LEAD_NAMES[sorted_idx[0]]}'),
    Patch(facecolor='#e67e22', edgecolor='#2c3e50', label=f'2nd: {LEAD_NAMES[sorted_idx[1]]}'),
    Patch(facecolor='#f1c40f', edgecolor='#2c3e50', label=f'3rd: {LEAD_NAMES[sorted_idx[2]]}'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/lead_importance_bar_chart.png',
            dpi=300, bbox_inches='tight')
print(f"    ✅ lead_importance_bar_chart.png")
plt.close()

# --- Save to CSV ---
importance_df = pd.DataFrame({
    'Lead': LEAD_NAMES,
    'Importance': avg_norm,
    'Std': std_norm,
    'Rank': [int(np.where(sorted_idx == i)[0][0]) + 1 for i in range(12)]
}).sort_values('Importance', ascending=False)

importance_df.to_csv(RESULTS_DIR + 'metrics/lead_importance_mamba.csv', index=False)
print(f"    ✅ lead_importance_mamba.csv")

print(f"\n  📊 Lead Importance Rankings (Mamba-2):")
print(f"  {'Rank':<6s} {'Lead':<6s} {'Importance':>12s} {'Std':>10s}")
print(f"  {'─'*36}")
for _, row in importance_df.iterrows():
    print(f"  {int(row['Rank']):<6d} {row['Lead']:<6s} {row['Importance']:>12.4f} {row['Std']:>10.4f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: TEMPORAL IMPORTANCE PROFILE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 6] Generating temporal importance profile...")

# Use the high-confidence MI case
key = 'MI_Correct_HighConf'
if key in attributions_dict:
    # Smooth temporal profile (same sigma as heatmaps for consistency)
    raw_temporal = np.abs(attributions_dict[key]).mean(axis=1)
    temporal_imp = gaussian_filter1d(raw_temporal, sigma=SMOOTH_SIGMA)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    t = np.arange(len(temporal_imp))

    # --- Attribution curve ---
    ax1.fill_between(t, temporal_imp, alpha=0.30, color='#c0392b', zorder=2)
    ax1.plot(t, temporal_imp, linewidth=2.2, color='#c0392b', zorder=3,
             label='Avg |Attribution|')

    # Peak marker
    peak_idx = np.argmax(temporal_imp)
    ax1.axvline(peak_idx, color='#2c3e50', linestyle='--', linewidth=1.5,
                label=f'Peak: sample {peak_idx} (~{peak_idx/100:.2f}s)', zorder=4)

    ax1.set_xlabel('Time (samples @ 100 Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg |Attribution|', fontsize=12, fontweight='bold',
                   color='#c0392b')
    ax1.tick_params(axis='y', labelcolor='#c0392b')

    # --- ECG waveform overlay (Lead II — most commonly used) ---
    ax2 = ax1.twinx()
    ecg_leadII = X_test[examples[key]][:, 1]  # Lead II
    ax2.plot(t, ecg_leadII, color='#7f8c8d', linewidth=0.9, alpha=0.6,
             zorder=1, label='Lead II (ECG)')
    ax2.set_ylabel('Lead II amplitude', fontsize=10, color='#7f8c8d')
    ax2.tick_params(axis='y', labelcolor='#7f8c8d')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

    ax1.set_title('Mamba-2: Temporal Importance Profile\n'
                  '(Smoothed, averaged across 12 leads — High-confidence MI case)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.set_xlim(0, len(temporal_imp))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR + 'figures/temporal_importance_profile.png',
                dpi=300, bbox_inches='tight')
    print(f"    ✅ temporal_importance_profile.png")
    plt.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("EXPLAINABILITY ANALYSIS COMPLETE — Bidirectional Mamba-2")
print(f"{'='*70}")

top3 = importance_df.head(3)['Lead'].values
print(f"""
📁 Generated Outputs:
   figures/
     • *_12lead_heatmap.png    — Full 12-lead ECGradCAM-style overlays
     • *_4lead_grid.png        — Selected leads (II, aVF, V2, V5) 2×2 grid
     • lead_importance_bar_chart.png
     • temporal_importance_profile.png
   metrics/
     • lead_importance_mamba.csv

📊 Key Findings:
   • Top 3 leads: {', '.join(top3)}
   • Most important: Lead {top3[0]} ({importance_df.iloc[0]['Importance']:.1%})
   • Peak temporal importance: sample {peak_idx} (~{peak_idx/100:.2f}s)

💡 Compare with CNN-LSTM:
   Load both lead_importance.csv (CNN-LSTM) and lead_importance_mamba.csv
   to compare lead rankings across architectures.

💾 All results in: {os.path.abspath(RESULTS_DIR)}
{'='*70}
""")