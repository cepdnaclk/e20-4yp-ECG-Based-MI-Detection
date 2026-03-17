#!/usr/bin/env python3
"""
============================================================================
Explainability Analysis: Integrated Gradients for Bidirectional Mamba-2
SUBCLASS-AWARE: AMI vs IMI Lead Importance Comparison
============================================================================

Supervisor requirement: XAI should be subclass-based, not random MI samples.
  - AMI (Anterior MI) → expect V1, V2, V3, V4 to dominate
  - IMI (Inferior MI) → expect II, III, aVF to dominate

For each subclass:
  - Pick 5 correctly classified pure-subclass ECGs
  - Compute IG attributions for each
  - Show 1 representative heatmap (highest confidence)
  - Show averaged lead importance across all 5

Final output: Side-by-side AMI vs IMI comparison figure

Dataset: A (Certain MI only) — Mamba-2 model
Style: ECGradCAM (Hicks et al., 2021)
============================================================================

Location:
  Mamba2_DatasetA_ValTest_Pure/scripts/7_xai_subclass_ami_vs_imi.py

Author : Chamath (E20342) — FYP ECG-IHD Detection
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from captum.attr import IntegratedGradients
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DATA_DIR = '../../cnn_lstm_datasetA_certain_mi/data/'
MODEL_PATH = '../models/datasetA_best.pt'
SUBCLASS_CSV = '../data/ptbxl_dataset_test_certain_vs_normal_with_mi_subclass.csv'
RESULTS_DIR = '../results/xai_subclass/'

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

IG_STEPS = 50
N_SAMPLES_PER_SUBCLASS = 5   # 5 ECGs per subclass for averaging
SMOOTH_SIGMA = 12             # Gaussian smoothing for heatmaps

JET = plt.cm.jet

os.makedirs(RESULTS_DIR + 'figures/', exist_ok=True)
os.makedirs(RESULTS_DIR + 'metrics/', exist_ok=True)

print(f"\n{'='*70}")
print("SUBCLASS-AWARE XAI — AMI vs IMI")
print("Model: Bidirectional Mamba-2 | Dataset A (Certain MI)")
print(f"{'='*70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL DEFINITION (self-contained)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class CFG:
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

try:
    from mamba_ssm import Mamba, Mamba2
    print(f"  ✅ mamba_ssm available")
except ImportError:
    print(f"  ❌ mamba_ssm not found"); sys.exit(1)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight.view(1,1,-1)

class GatedMLP(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        hidden = expansion * d_model
        self.fc1 = nn.Linear(d_model, 2*hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        u, v = self.fc1(x).chunk(2, dim=-1)
        return self.drop(self.fc2(F.silu(u) * v))

class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)

class MambaMixer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, model_type="mamba"):
        super().__init__()
        Cls = Mamba2 if model_type == "mamba2" else Mamba
        self.mixer = Cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x):
        return self.mixer(x)

class BiMixer(nn.Module):
    def __init__(self, mixer_fwd, mixer_bwd=None, d_model=256, mode="concat"):
        super().__init__()
        self.fwd = mixer_fwd
        self.bwd = mixer_bwd if mixer_bwd is not None else mixer_fwd
        self.mode = mode
        self.proj = nn.Linear(2*d_model, d_model) if mode == "concat" else None
    def forward(self, x):
        y_f = self.fwd(x)
        y_b = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        if self.mode == "concat": return self.proj(torch.cat([y_f, y_b], dim=-1))
        return 0.5*(y_f+y_b) if self.mode=="avg" else y_f+y_b

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, model_type, bidirectional, mlp_expand=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        mixer = MambaMixer(d_model, d_state, d_conv, expand, model_type)
        if bidirectional:
            self.mixer = BiMixer(mixer, MambaMixer(d_model, d_state, d_conv, expand, model_type), d_model, "concat")
        else:
            self.mixer = mixer
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, mlp_expand, dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.drop1(self.mixer(self.norm1(x)))
        return x + self.drop2(self.mlp(self.norm2(x)))

class ECGMambaClassifier(nn.Module):
    def __init__(self, in_ch=12):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, cfg.D_MODEL)
        self.drop_in = nn.Dropout(cfg.DROPOUT)
        self.blocks = nn.ModuleList([
            MambaBlock(cfg.D_MODEL, cfg.D_STATE, cfg.D_CONV, cfg.EXPAND,
                       cfg.DROPOUT, cfg.MODEL_TYPE, cfg.BIDIRECTIONAL)
            for _ in range(cfg.N_LAYERS)])
        self.norm_out = RMSNorm(cfg.D_MODEL)
        self.pool = AttnPool(cfg.D_MODEL) if cfg.POOLING == "attn" else None
        self.head = nn.Linear(cfg.D_MODEL, 1)
    def forward(self, x):
        x = self.drop_in(self.in_proj(x.transpose(1,2)))
        for blk in self.blocks: x = blk(x)
        x = self.norm_out(x)
        return self.head(x.mean(1) if self.pool is None else self.pool(x)).squeeze(-1)

class MambaIGWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.BatchNorm1d, RMSNorm)): m.eval()
        logits = self.model(x)
        self.model.eval()
        return torch.sigmoid(logits)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD MODEL, DATA, AND SUBCLASS LABELS                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 1] Loading model, data, and subclass labels...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = ECGMambaClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print(f"  ✅ Model loaded: {MODEL_PATH}")

X_test = np.load(DATA_DIR + 'X_test.npy')        # (N, 1000, 12)
y_test = np.load(DATA_DIR + 'y_test.npy')

# Load subclass CSV and build ecg_id → index mapping
df_sub = pd.read_csv(SUBCLASS_CSV)
test_ecg_ids = df_sub['ecg_id'].values  # ordered same as X_test

# Build ecg_id → test array index
ecg_id_to_idx = {eid: i for i, eid in enumerate(test_ecg_ids)}

# Get model predictions
X_torch = torch.FloatTensor(X_test.transpose(0, 2, 1)).to(device)
with torch.no_grad():
    logits = model(X_torch).cpu().numpy()
    y_proba = 1.0 / (1.0 + np.exp(-logits))
y_pred = (y_proba >= 0.5).astype(int)

print(f"  ✅ Test data: {X_test.shape}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: SELECT PURE AMI AND IMI SAMPLES                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 2] Selecting pure AMI and IMI samples...")


def select_subclass_samples(subclass_name, n=N_SAMPLES_PER_SUBCLASS):
    """Select top-n highest confidence correctly classified samples for a subclass."""
    # Pure subclass only (exclude mixed like "AMI; IMI")
    mask = df_sub['mi_diagnostic_subclass'] == subclass_name
    sub_ecg_ids = df_sub.loc[mask, 'ecg_id'].values

    # Find correctly classified ones
    candidates = []
    for eid in sub_ecg_ids:
        if eid not in ecg_id_to_idx:
            continue
        idx = ecg_id_to_idx[eid]
        if y_test[idx] == 1 and y_pred[idx] == 1:  # true MI, predicted MI
            candidates.append((idx, y_proba[idx], eid))

    # Sort by confidence (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = candidates[:n]
    print(f"  {subclass_name}: {len(candidates)} correct, selected top {len(selected)}")
    for idx, prob, eid in selected:
        print(f"    ecg_id={eid:>6d}  prob={prob:.1%}")

    return selected  # list of (array_idx, probability, ecg_id)


ami_samples = select_subclass_samples('AMI')
imi_samples = select_subclass_samples('IMI')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: COMPUTE ATTRIBUTIONS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 3] Computing Integrated Gradients...")

model_wrapper = MambaIGWrapper(model)
ig = IntegratedGradients(model_wrapper)


def compute_attributions(ecg_sample):
    inp = torch.FloatTensor(ecg_sample.T).unsqueeze(0).to(device)
    inp.requires_grad = True
    baseline = torch.zeros_like(inp).to(device)
    attrs = ig.attribute(inp, baselines=baseline, n_steps=IG_STEPS)
    return attrs.squeeze(0).permute(1, 0).cpu().detach().numpy()  # (1000, 12)


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
    """Average normalized lead importance across multiple samples."""
    importances = []
    for attrs in attrs_list:
        lead_imp = np.abs(attrs).sum(axis=0)  # sum |attr| over time
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

# Print rankings
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

# Value labels
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

fig.suptitle('Mamba-2: Lead Importance by MI Subclass\n'
             f'(Dataset A — {N_SAMPLES_PER_SUBCLASS} samples per subclass)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/lead_importance_ami_vs_imi.png',
            dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: lead_importance_ami_vs_imi.png")
plt.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: REPRESENTATIVE HEATMAPS (best AMI + best IMI, side by side)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print(f"\n[Step 5] Creating representative heatmaps...")


def plot_ecg_heatmap_panel(ax, ecg_signal, attr_smooth, lead_name,
                           global_vmax=None, show_xlabel=False):
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


# --- Side-by-side 12-lead: AMI (left) vs IMI (right) ---
best_ami_idx = ami_samples[0][0]
best_ami_prob = ami_samples[0][1]
best_ami_eid = ami_samples[0][2]
best_ami_attrs = smooth_attributions(ami_attrs[0])

best_imi_idx = imi_samples[0][0]
best_imi_prob = imi_samples[0][1]
best_imi_eid = imi_samples[0][2]
best_imi_attrs = smooth_attributions(imi_attrs[0])

# Global vmax across BOTH for fair comparison
global_vmax = max(best_ami_attrs.max(), best_imi_attrs.max())

fig, axes = plt.subplots(12, 2, figsize=(20, 20), facecolor='white')
fig.suptitle(
    f'Mamba-2 Integrated Gradients — AMI vs IMI Comparison\n'
    f'AMI (ecg {best_ami_eid}, {best_ami_prob:.1%})                    '
    f'IMI (ecg {best_imi_eid}, {best_imi_prob:.1%})',
    fontsize=15, fontweight='bold', y=0.995
)

for i in range(12):
    # AMI (left column)
    plot_ecg_heatmap_panel(
        axes[i, 0], X_test[best_ami_idx][:, i], best_ami_attrs[:, i],
        LEAD_NAMES[i], global_vmax=global_vmax, show_xlabel=(i == 11))
    # IMI (right column)
    plot_ecg_heatmap_panel(
        axes[i, 1], X_test[best_imi_idx][:, i], best_imi_attrs[:, i],
        LEAD_NAMES[i], global_vmax=global_vmax, show_xlabel=(i == 11))

# Column titles
axes[0, 0].set_title('AMI (Anterior MI)', fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_title('IMI (Inferior MI)', fontsize=13, fontweight='bold', pad=10)

# Colorbar
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


# --- 4-lead comparison (2×2 per subclass → 2×4 total) ---
# AMI-relevant leads: V1, V2, V3, V4
AMI_LEADS = [6, 7, 8, 9]    # V1, V2, V3, V4
# IMI-relevant leads: II, III, aVF, V6
IMI_LEADS = [1, 2, 5, 11]   # II, III, aVF, V6

fig, axes = plt.subplots(2, 4, figsize=(22, 8), facecolor='white')
fig.suptitle(
    f'Mamba-2: Clinically Relevant Leads — AMI vs IMI\n'
    f'AMI (ecg {best_ami_eid}, {best_ami_prob:.1%})  |  '
    f'IMI (ecg {best_imi_eid}, {best_imi_prob:.1%})',
    fontsize=14, fontweight='bold'
)

# Row 0: AMI with its relevant leads (V1-V4)
for i, lead_idx in enumerate(AMI_LEADS):
    plot_ecg_heatmap_panel(
        axes[0, i], X_test[best_ami_idx][:, lead_idx], best_ami_attrs[:, lead_idx],
        f'Lead {LEAD_NAMES[lead_idx]}', global_vmax=global_vmax, show_xlabel=False)
    axes[0, i].set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')

# Row 1: IMI with its relevant leads (II, III, aVF, V6)
for i, lead_idx in enumerate(IMI_LEADS):
    plot_ecg_heatmap_panel(
        axes[1, i], X_test[best_imi_idx][:, lead_idx], best_imi_attrs[:, lead_idx],
        f'Lead {LEAD_NAMES[lead_idx]}', global_vmax=global_vmax, show_xlabel=True)
    axes[1, i].set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')

# Row labels
axes[0, 0].annotate('AMI', xy=(-0.35, 0.5), xycoords='axes fraction',
                     fontsize=14, fontweight='bold', color='#c0392b',
                     ha='center', va='center', rotation=90)
axes[1, 0].annotate('IMI', xy=(-0.35, 0.5), xycoords='axes fraction',
                     fontsize=14, fontweight='bold', color='#2980b9',
                     ha='center', va='center', rotation=90)

# Colorbar
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
print("SUBCLASS XAI COMPLETE")
print(f"{'='*70}")
print(f"""
📊 Lead Importance:
   AMI top 3: {', '.join(ami_top3)}   (expected: V1, V2, V3 — anterior leads)
   IMI top 3: {', '.join(imi_top3)}   (expected: II, III, aVF — inferior leads)

📁 Outputs:
   figures/
     • lead_importance_ami_vs_imi.png       — Side-by-side bar chart
     • heatmap_ami_vs_imi_12lead.png        — Full 12-lead comparison
     • heatmap_ami_vs_imi_relevant_leads.png — Clinically relevant leads only
   metrics/
     • lead_importance_ami_vs_imi.csv

💡 Clinical Validation:
   If AMI → V1-V3 and IMI → II,III,aVF, the model learned
   real coronary territory anatomy, not spurious correlations.

{'='*70}
""")