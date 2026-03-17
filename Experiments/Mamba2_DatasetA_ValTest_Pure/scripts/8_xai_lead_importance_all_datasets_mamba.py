#!/usr/bin/env python3
"""
============================================================================
XAI Lead Importance: Bidirectional Mamba-2 — Datasets A, C, D
Outputs CSV only (no figures) — as requested by supervisor
============================================================================

Run from: Mamba2_DatasetA_ValTest_Pure/scripts/

Produces: results/xai_subclass/metrics/lead_importance_all_datasets.csv
============================================================================
"""

import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# All datasets share the same test data (loaded from Dataset A's dir)
TEST_DATA_DIR = '../../cnn_lstm_datasetA_certain_mi/data/'
SUBCLASS_CSV = '../data/ptbxl_dataset_test_certain_vs_normal_with_mi_subclass.csv'
RESULTS_DIR = '../results/xai_subclass/metrics/'

IG_STEPS = 50
N_SAMPLES = 5

# ── Dataset model paths ───────────────────────────────────────────────────
DATASETS = {
    'A': '../models/datasetA_best.pt',
    'C': '../../Mamba2_DatasetC_ValTest_Pure/models/datasetC_best.pt',
    'D': '../../Mamba2_DatasetD_ValTest_Pure/models/datasetd_best.pt',
}

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
except ImportError:
    print("❌ mamba_ssm not found"); sys.exit(1)


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
# ║  HELPERS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Load test data and subclass CSV once (shared across all datasets)
X_test = np.load(TEST_DATA_DIR + 'X_test.npy')    # (N, 1000, 12)
y_test = np.load(TEST_DATA_DIR + 'y_test.npy')
df_sub = pd.read_csv(SUBCLASS_CSV)
test_ecg_ids = df_sub['ecg_id'].values
ecg_id_to_idx = {eid: i for i, eid in enumerate(test_ecg_ids)}


def select_samples(subclass_name, y_proba, y_pred, n=N_SAMPLES):
    mask = df_sub['mi_diagnostic_subclass'] == subclass_name
    sub_ecg_ids = df_sub.loc[mask, 'ecg_id'].values
    candidates = []
    for eid in sub_ecg_ids:
        if eid not in ecg_id_to_idx:
            continue
        idx = ecg_id_to_idx[eid]
        if y_test[idx] == 1 and y_pred[idx] == 1:
            candidates.append((idx, y_proba[idx], eid))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def compute_ig(wrapper, ig, ecg_sample):
    # Mamba expects (B, 12, 1000) — transpose from (1000, 12)
    inp = torch.FloatTensor(ecg_sample.T).unsqueeze(0).to(device)
    inp.requires_grad = True
    baseline = torch.zeros_like(inp).to(device)
    attrs = ig.attribute(inp, baselines=baseline, n_steps=IG_STEPS)
    return attrs.squeeze(0).permute(1, 0).cpu().detach().numpy()  # (1000, 12)


def compute_lead_importance(attrs_list):
    importances = []
    for attrs in attrs_list:
        lead_imp = np.abs(attrs).sum(axis=0)
        lead_imp_norm = lead_imp / lead_imp.sum()
        importances.append(lead_imp_norm)
    importances = np.array(importances)
    return importances.mean(axis=0), importances.std(axis=0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

all_results = []

print(f"\n{'='*70}")
print("Mamba-2 XAI — Lead Importance for Datasets A, C, D")
print(f"Device: {device}")
print(f"{'='*70}")

for ds_name, model_path in DATASETS.items():
    print(f"\n{'─'*50}")
    print(f"  Dataset {ds_name}")
    print(f"{'─'*50}")

    # Load model
    model = ECGMambaClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"  ✅ Model: {model_path}")

    # Predictions — Mamba outputs raw logits → apply sigmoid
    X_torch = torch.FloatTensor(X_test.transpose(0, 2, 1)).to(device)  # (N, 12, 1000)
    with torch.no_grad():
        logits = model(X_torch).cpu().numpy()
        y_proba = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (y_proba >= 0.5).astype(int)

    # Setup IG
    wrapper = MambaIGWrapper(model)
    ig = IntegratedGradients(wrapper)

    for subclass in ['AMI', 'IMI']:
        samples = select_samples(subclass, y_proba, y_pred)
        print(f"  {subclass}: {len(samples)} samples selected")

        if len(samples) == 0:
            print(f"    ⚠ No valid samples — skipping")
            for lead in LEAD_NAMES:
                all_results.append({
                    'Dataset': ds_name, 'Subclass': subclass,
                    'Lead': lead, 'Importance': float('nan'), 'Std': float('nan')
                })
            continue

        attrs_list = []
        for idx, prob, eid in samples:
            print(f"    ecg_id={eid:>6d}  prob={prob:.1%}")
            attrs_list.append(compute_ig(wrapper, ig, X_test[idx]))

        avg, std = compute_lead_importance(attrs_list)
        ranked = np.argsort(avg)[::-1]
        top3 = [LEAD_NAMES[i] for i in ranked[:3]]
        print(f"    → Top 3: {', '.join(top3)}")

        for i, lead in enumerate(LEAD_NAMES):
            all_results.append({
                'Dataset': ds_name, 'Subclass': subclass,
                'Lead': lead, 'Importance': avg[i], 'Std': std[i]
            })

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SAVE COMBINED CSV                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

results_df = pd.DataFrame(all_results)
out_path = RESULTS_DIR + 'lead_importance_all_datasets.csv'
results_df.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")

# ── Print summary table ───────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUMMARY — Top 3 Leads per Dataset × Subclass (Mamba-2)")
print(f"{'='*70}")
print(f"{'Dataset':<10} {'Subclass':<8} {'#1':<6} {'#2':<6} {'#3':<6}")
print(f"{'─'*40}")

for ds in ['A', 'C', 'D']:
    for sc in ['AMI', 'IMI']:
        sub = results_df[(results_df['Dataset'] == ds) & (results_df['Subclass'] == sc)]
        if sub['Importance'].isna().all():
            print(f"{ds:<10} {sc:<8} {'N/A':<6} {'N/A':<6} {'N/A':<6}")
            continue
        top3_idx = sub.nlargest(3, 'Importance')['Lead'].values
        print(f"{ds:<10} {sc:<8} {top3_idx[0]:<6} {top3_idx[1]:<6} {top3_idx[2]:<6}")

print(f"\n{'='*70}")
print(f"Expected:  AMI → V1, V2, V3    IMI → II, III, aVF")
print(f"{'='*70}")