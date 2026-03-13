# Quality Over Quantity: The Impact of Diagnostic Certainty of Data in Deep Learning for ECG Analysis

> **Final Year Project — BSc (Hons) Computer Engineering, University of Peradeniya**

This repository contains the code, dataset configurations, and trained models for our research on how **diagnostic label certainty** impacts deep learning model performance, calibration, and clinical interpretability in ECG-based **Myocardial Infarction (MI) detection**.

**Core Finding:** High-confidence, cardiologist-validated labels (quality) consistently outperform larger but noisier datasets (quantity) — across both architectures and all evaluation metrics.

---

## Table of Contents

- [Abstract](#abstract)
- [Project Overview](#project-overview)
- [Dataset Design](#dataset-design)
- [Architectures](#architectures)
- [Key Results](#key-results)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Abstract

Cardiovascular diseases remain the leading cause of death globally, with Myocardial Infarction (MI) being a critical subset. While deep learning for automated ECG interpretation has advanced rapidly, most studies overlook a fundamental question: **does the certainty of diagnostic labels matter more than the volume of training data?**

We investigate this by training two architectures — a **CNN-LSTM hybrid** and a **Bidirectional Mamba-2** — on three carefully constructed dataset variants from PTB-XL, each with identical normal cases but varying MI label certainty. Our ablation study demonstrates that models trained on 100% confidence, human-validated MI labels achieve superior discrimination (AUROC), better calibration (ECE, Brier Score), and more clinically interpretable XAI outputs — even with fewer training samples.

---

## Project Overview

### Research Question

> Does diagnostic label certainty matter more than dataset size for training trustworthy medical AI?

### Approach

We designed an **ablation-style experiment** isolating the effect of label certainty while controlling for architecture, preprocessing, and evaluation methodology:

- **Binary classification**: MI vs. Normal (12-lead ECG)
- **Dataset**: PTB-XL v1.0.3 (100 Hz)
- **Patient-wise splitting**: Prevents data leakage (train/val/test have zero patient overlap)
- **Same validation and test sets** across all dataset variants for fair comparison

---

## Dataset Design

All three training sets share the **same 4,451 pure normal ECGs** (NORM with 100% confidence, no MI at any probability). They differ only in which MI cases are included:

| Dataset | MI Cases | Description | Total Train |
|---------|----------|-------------|-------------|
| **A** (Certain MI) | 1,194 | Only MI labels with **100% confidence**, human-validated by cardiologists | 5,645 |
| **C** (All MI) | 2,387 | All MI cases — both certain (100%) and uncertain (<100%) | 6,838 |
| **D** (Uncertain MI) | 1,193 | Only MI labels with **<100% confidence** (uncertain/ambiguous cases) | 5,644 |

**Shared evaluation sets:**
- Validation: 1,234 records
- Test: 1,253 records

### Filtering Criteria

**Normal cases (label = 0):**
- NORM SCP code with 100% confidence
- No MI diagnosis at any probability level
- No other cardiac conditions (STTC, HYP, CD)

**Certain MI cases (label = 1, Dataset A):**
- At least one MI SCP code with 100% confidence
- Human-validated by cardiologists (`validated_by_human = True`)

**Uncertain MI cases (label = 1, Dataset D):**
- MI SCP codes present but all at <100% confidence

---

## Architectures

### 1. CNN-LSTM Hybrid

A temporal feature extraction pipeline combining convolutional and recurrent layers:

- **Input**: (batch, 1000, 12) — 10s ECG at 100 Hz, 12 leads
- **CNN Block**: 3× Conv1D layers (64→128→256 filters) with BatchNorm, ReLU, MaxPool, Dropout
- **LSTM Block**: 2-layer Bidirectional LSTM (128 hidden units)
- **Classifier**: FC layers with dropout → sigmoid output
- **Parameters**: ~367K trainable
- **Framework**: PyTorch
- **XAI Method**: Grad-CAM (on final Conv1D layer)

### 2. Bidirectional Mamba-2

A state space model (SSM) architecture for efficient long-range temporal modeling:

- **Input**: (batch, 1000, 12) — same as CNN-LSTM
- **Architecture**: Bidirectional Mamba-2 blocks with selective gating
- **Advantage**: Linear complexity with sequence length (vs. quadratic for Transformers)
- **Framework**: PyTorch
- **XAI Method**: Integrated Gradients (via Captum library — Grad-CAM not applicable due to absence of convolutional layers)

---

## Key Results

### Consistent Ranking: A > C > D

Across **both architectures**, Dataset A (100% confidence labels) consistently achieves:
- Highest AUROC and accuracy
- Lowest calibration error (ECE)
- Best Brier scores
- Most clinically interpretable XAI outputs

This confirms our central thesis: **label quality > quantity**.

### CNN-LSTM Baseline (Dataset A)

| Metric | Value |
|--------|-------|
| Accuracy | 95.87% |
| Precision | 88.76% |
| Recall (Sensitivity) | 93.31% |
| Specificity | 96.60% |
| F1-Score | 90.98% |
| ROC-AUC | 99.06% |

### Notable Finding: Calibration Degradation

The Mamba-2 model trained on **Dataset D** (uncertain labels) showed significantly degraded calibration (high ECE), while Dataset A maintained low ECE — reinforcing that uncertain labels harm not just accuracy but model **trustworthiness**.

### Model Uncertainty Mirrors Human Uncertainty

When cardiologists expressed uncertainty (lower confidence scores), the model's prediction confidence appropriately decreased — demonstrating a positive correlation between human and model confidence. This is clinically meaningful for flagging ambiguous cases that need expert review.

---

## Explainable AI (XAI)

XAI analysis was performed across **all three datasets** for both architectures:

### CNN-LSTM: Grad-CAM
- Activation heatmaps from the final Conv1D layer overlaid on 12-lead ECG signals
- **Dataset A** models focus on physiologically correct leads: V1–V4 for anterior MI, leads II/III/aVF for inferior MI
- **Dataset D** models lose clinical interpretability — diffuse, non-specific activations

### Mamba-2: Integrated Gradients (Captum)
- Lead importance rankings: **V2, Lead II, V3** consistently ranked as top contributing leads
- Aligns with cardiac anatomy (LAD territory coverage)
- Cross-dataset comparison shows Dataset A produces the most clinically meaningful attribution maps

---

## Repository Structure

```
├── data/
│   ├── ptbxl_datasetA_train.csv          # Dataset A: Certain MI + Pure Normal
│   ├── ptbxl_datasetB_train.csv          # Dataset D: Uncertain MI + Pure Normal
│   ├── ptbxl_datasetC_train.csv          # Dataset C: All MI + Pure Normal
│   ├── ptbxl_dataset_val.csv             # Shared validation set
│   ├── ptbxl_dataset_test.csv            # Shared test set
│   └── ptbxl_ecg_labels_mi_vs_normal_scp100_human.csv  # Full filtered label file
├── models/
│   ├── cnn_lstm/                         # CNN-LSTM training & evaluation
│   └── mamba2/                           # Bidirectional Mamba-2 training & evaluation
├── xai/
│   ├── gradcam/                          # Grad-CAM analysis (CNN-LSTM)
│   └── integrated_gradients/             # Integrated Gradients analysis (Mamba-2)
├── preprocessing/
│   ├── dataset_filtering.py              # PTB-XL filtering & label extraction
│   └── signal_processing.py             # Baseline wander removal, standardization
├── evaluation/
│   ├── evaluate.py                       # Metrics: Accuracy, AUROC, ECE, Brier
│   └── calibration_analysis.py           # Reliability diagrams & ECE computation
├── notebooks/
│   └── Mamba_Filtered.ipynb              # Mamba-2 experiment notebook
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (GPU required for training)
- PTB-XL dataset v1.0.3

### Install Dependencies

```bash
git clone https://github.com/<your-username>/ecg-mi-label-certainty.git
cd ecg-mi-label-certainty
pip install -r requirements.txt
```

### Key Libraries

| Library | Purpose |
|---------|---------|
| PyTorch 2.5+ | Model training (both architectures) |
| Captum | Integrated Gradients (Mamba-2 XAI) |
| WFDB | Reading PTB-XL ECG files |
| Scikit-learn | Metrics, preprocessing, calibration |
| NumPy, Pandas | Data processing |
| Matplotlib, Seaborn | Visualization |
| SciPy | Signal filtering (baseline wander removal at 0.5 Hz) |

### Download PTB-XL

```bash
# Download PTB-XL v1.0.3 from PhysioNet
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Place the dataset so that `records100/` and `records500/` directories are accessible from your project root.

---

## Usage

### 1. Preprocessing & Dataset Generation

```bash
python preprocessing/dataset_filtering.py --ptbxl_path /path/to/ptb-xl/
```

This generates the three dataset CSVs (A, C, D) with patient-wise splits.

### 2. Training

```bash
# Train CNN-LSTM on Dataset A
python models/cnn_lstm/train.py --dataset A --epochs 50 --batch_size 32

# Train Mamba-2 on Dataset A
python models/mamba2/train.py --dataset A --epochs 50 --batch_size 32
```

### 3. Evaluation

```bash
python evaluation/evaluate.py --model cnn_lstm --dataset A --checkpoint models/cnn_lstm/best_model.pth
```

### 4. XAI Analysis

```bash
# Grad-CAM for CNN-LSTM
python xai/gradcam/run_gradcam.py --model cnn_lstm --dataset A

# Integrated Gradients for Mamba-2
python xai/integrated_gradients/run_ig.py --model mamba2 --dataset A
```

---

## Team

| Name | Registration | Role |
|------|-------------|------|
| Chamath Rupasinghe | E/20/342 | CNN-LSTM, Dataset Design, XAI |
| M.L. De Croos Rubin | E/20/054 | Bidirectional Mamba-2 |
| S.M.N.N. Padeniya | E/20/276 | Evaluation, Calibration Analysis |

### Supervisors

- **Dr. Isuru Nawinne** — Department of Computer Engineering, University of Peradeniya
- **Dr. Isuri Devindi** — Department of Computer Engineering, University of Peradeniya
- **Dr. Jørgen Kanters** — Cardiologist (Clinical Advisor)

---

## Acknowledgements

- [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/) — Wagner et al., 2020
- [PTB-XL Benchmarking Repository](https://github.com/helme/ecg_ptbxl_benchmarking) — Strodthoff et al., 2021
- [Captum Library](https://captum.ai/) — PyTorch model interpretability
- University of Peradeniya GPU Cluster (`ada.ce.pdn.ac.lk`)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{rupasinghe2026qualityoverquantity,
  title={Quality Over Quantity: The Impact of Diagnostic Certainty of Data in Deep Learning for ECG Analysis},
  author={Rupasinghe, Chamath and De Croos Rubin, M.L. and Padeniya, S.M.N.N.},
  year={2026},
  institution={University of Peradeniya}
}
```

---

## License

This project is for academic and research purposes. The PTB-XL dataset is available under the [PhysioNet Credentialed Health Data License](https://physionet.org/content/ptb-xl/1.0.3/).
