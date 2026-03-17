#!/usr/bin/env python3
"""
============================================================================
XAI Lead Importance: CNN-LSTM — Datasets A, C, D
Outputs CSV only (no figures) — as requested by supervisor
============================================================================

Run from: cnn_lstm_datasetA_certain_mi/scripts/
          (or adjust BASE_DIR below)

Produces: results/xai_subclass/metrics/lead_importance_all_datasets.csv
============================================================================
"""

import sys
sys.path.append('../../shared_utils')

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from model import CNNLSTM

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

SUBCLASS_CSV = '../data/ptbxl_dataset_test_certain_vs_normal_with_mi_subclass.csv'
RESULTS_DIR = '../results/xai_subclass/metrics/'

IG_STEPS = 50
N_SAMPLES = 5

# ── Dataset configs ───────────────────────────────────────────────────────
# Each entry: (label, data_dir, model_path)
# Adjust paths if your directory structure differs

DATASETS = {
    'A': {
        'data_dir': '../../cnn_lstm_datasetA_certain_mi/data/',
        'model_path': '../../cnn_lstm_datasetA_certain_mi/saved_models/best_model.pth',
    },
    'C': {
        'data_dir': '../../cnn_lstm_datasetC_all_mi/data/',
        'model_path': '../../cnn_lstm_datasetC_all_mi/saved_models/best_model.pth',
    },
    'D': {
        'data_dir': '../../cnn_lstm_datasetD_balanced/data/',
        'model_path': '../../cnn_lstm_datasetD_balanced/saved_models/best_model.pth',
    },
}

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load subclass CSV once (same test set for all)
df_sub = pd.read_csv(SUBCLASS_CSV)
test_ecg_ids = df_sub['ecg_id'].values
ecg_id_to_idx = {eid: i for i, eid in enumerate(test_ecg_ids)}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WRAPPER                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
                m.eval()
        out = self.model(x)
        self.model.eval()
        return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def select_samples(subclass_name, y_test, y_proba, y_pred, n=N_SAMPLES):
    mask = df_sub['mi_diagnostic_subclass'] == subclass_name
    sub_ecg_ids = df_sub.loc[mask, 'ecg_id'].values

    candidates = []
    for eid in sub_ecg_ids:
        if eid not in ecg_id_to_idx:
            continue
        idx = ecg_id_to_idx[eid]
        if y_test[idx] == 1 and y_pred[idx] == 1:
            candidates.append((idx, y_proba[idx], eid))

    # Filter sigmoid saturation
    candidates = [(idx, p, eid) for idx, p, eid in candidates if 0.85 <= p <= 0.99]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def compute_ig(model_wrapper, ig, ecg_sample, device):
    inp = torch.FloatTensor(ecg_sample).unsqueeze(0).to(device)
    inp.requires_grad = True
    baseline = torch.zeros_like(inp).to(device)
    attrs = ig.attribute(inp, baselines=baseline, n_steps=IG_STEPS)
    return attrs.squeeze(0).cpu().detach().numpy()


def compute_lead_importance(attrs_list):
    importances = []
    for attrs in attrs_list:
        lead_imp = np.abs(attrs).sum(axis=0)
        lead_imp_norm = lead_imp / lead_imp.sum()
        importances.append(lead_imp_norm)
    importances = np.array(importances)
    return importances.mean(axis=0), importances.std(axis=0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LOOP — ALL DATASETS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

all_results = []

print(f"\n{'='*70}")
print("CNN-LSTM XAI — Lead Importance for Datasets A, C, D")
print(f"Device: {device}")
print(f"{'='*70}")

for ds_name, ds_cfg in DATASETS.items():
    print(f"\n{'─'*50}")
    print(f"  Dataset {ds_name}")
    print(f"{'─'*50}")

    # Load model
    model = CNNLSTM()
    ckpt = torch.load(ds_cfg['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  ✅ Model: {ds_cfg['model_path']}")

    # Load test data (same test set, but loaded from each experiment's data dir)
    X_test = np.load(ds_cfg['data_dir'] + 'X_test.npy')
    y_test = np.load(ds_cfg['data_dir'] + 'y_test.npy')

    # Predictions
    with torch.no_grad():
        y_proba = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    y_pred = (y_proba >= 0.5).astype(int)

    # Setup IG
    wrapper = ModelWrapper(model)
    ig = IntegratedGradients(wrapper)

    for subclass in ['AMI', 'IMI']:
        samples = select_samples(subclass, y_test, y_proba, y_pred)
        print(f"  {subclass}: {len(samples)} samples selected")

        if len(samples) == 0:
            print(f"    ⚠ No valid samples — skipping")
            for i, lead in enumerate(LEAD_NAMES):
                all_results.append({
                    'Dataset': ds_name, 'Subclass': subclass,
                    'Lead': lead, 'Importance': float('nan'), 'Std': float('nan')
                })
            continue

        # Compute attributions
        attrs_list = []
        for idx, prob, eid in samples:
            print(f"    ecg_id={eid:>6d}  prob={prob:.1%}")
            attrs_list.append(compute_ig(wrapper, ig, X_test[idx], device))

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
print("SUMMARY — Top 3 Leads per Dataset × Subclass")
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