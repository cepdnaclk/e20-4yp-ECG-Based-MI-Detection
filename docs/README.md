---
layout: home
permalink: index.html
repository-name: e20-4yp-ECG-Based-MI-Detection
title: "Quality Over Quantity: The Impact of Diagnostic Certainty of Data in Deep Learning for ECG Analysis"
---

# Quality Over Quantity: The Impact of Diagnostic Certainty of Data in Deep Learning for ECG Analysis

#### Team

- E/20/054, Mineth De Croos, [e20054@eng.pdn.ac.lk](mailto:e20054@eng.pdn.ac.lk) (Bidirectional Mamba-2)
- E/20/276, Nisitha Padeniya, [e20276@eng.pdn.ac.lk](mailto:e20276@eng.pdn.ac.lk) (Evaluation, Dataset Design, XAI)
- E/20/342, Chamath Rupasinghe, [e20342@eng.pdn.ac.lk](mailto:e20342@eng.pdn.ac.lk) (CNN-LSTM, XAI)

#### Supervisors

- Dr. Isuru Nawinne, [isurunawinne@eng.pdn.ac.lk](mailto:isurunawinne@eng.pdn.ac.lk)
- Isuri Devindi, [isurid@eng.pdn.ac.lk](mailto:isurid@eng.pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

Deep learning models for ECG-based myocardial infarction (MI) detection often prioritize discriminative accuracy over probability calibration, overlooking how the diagnostic certainty of training data impacts clinical reliability. We trained a CNN-LSTM and a Bidirectional Mamba-2 architecture on the PTB-XL dataset using three distinct cohorts: exclusively certain cases, all inclusive cases, and a mixed cohort matched for size. When evaluated against a standardized test set of exclusively certain cases, models trained on high-certainty labels consistently yielded the most accurate calibration, achieving Expected Calibration Errors of 0.0184 and 0.0339 respectively. Conversely, substituting certain data with uncertain labels in the size-matched mixed cohort significantly increased calibration error and false positive rates. Subclass-aware Integrated Gradients analysis confirmed that models trained on certain data learned physiologically congruent lead importance patterns, demonstrating that label quality, rather than sheer data quantity, fundamentally drives both model trustworthiness and clinical interpretability in medical AI.

## Related works

Deep learning has reached cardiologist-level performance in automated ECG interpretation, and the PTB-XL benchmark standardized evaluation across more than 21,000 ECGs. However, ECG studies commonly report only discrimination metrics (accuracy, AUROC) and rarely assess whether predicted probabilities are reliable. Work on imperfect supervision suggests cleaner, higher-confidence labels improve robustness, but this had not been examined for ECG labels that carry clinically graded certainty. State-space models such as Mamba and Mamba-2 perform strongly on long sequences, but their use for ECG classification remains limited.

This work addresses three gaps: (1) the absence of calibration metrics in ECG AI evaluation, (2) the lack of controlled experiments separating label certainty from dataset size, and (3) the need for cross-architecture validation of such findings.

## Methodology

**Data.** We use PTB-XL (21,799 clinical 12-lead ECGs, 10 s, from 18,869 patients), keeping only records whose diagnostic statements were validated by a cardiologist (`validated_by_human`). Using the SCP-code confidence scores we define three groups:

| Group | Definition | ECGs | Patients |
|---|---|---|---|
| Pure Normal | NORM at 100% confidence, no other diagnostic statements | 5,848 | 5,601 |
| Certain MI | Maximum MI confidence = 100% | 1,745 | 1,557 |
| Uncertain MI | MI present, maximum MI confidence < 100% | 1,732 | 1,634 |

**Splits.** Official PTB-XL stratified folds give patient-wise splits (folds 1–8 train, 9 validation, 10 test). Uncertain MI is removed from validation and test, so every model is evaluated only on diagnostically certain labels.

**Training variants.** The validation (977 ECGs) and test (971 ECGs) sets are identical for every variant; only the training set changes.

| Variant | Pure Normal | Certain MI | Uncertain MI | Role |
|---|---|---|---|---|
| A – Certain MI only | 4,451 | 1,194 | 0 | Primary |
| B – Uncertain MI only | 4,451 | 0 | 1,193 | Exploratory |
| C – All MI | 4,451 | 1,194 | 1,193 | Maximal data |
| D – Size-matched mix | 4,451 | 597 | 597 | Primary (same size as A) |
| Validation | 694 | 283 | 0 | Shared |
| Test | 703 | 268 | 0 | Shared |

The primary comparison is A vs. D, which isolates label certainty at a fixed training size.

**Preprocessing.** Signals at 100 Hz (1000 samples × 12 leads), baseline-wander removal with a 0.5 Hz Butterworth high-pass filter, and lead-wise Z-score normalization using training-set statistics.

## Experiment Setup and Implementation

| Setting | CNN-LSTM | Bidirectional Mamba-2 |
|---|---|---|
| Architecture | 3× Conv1D (16→32→64, kernel 5) with BatchNorm, ReLU, MaxPool, Dropout 0.3; BiLSTM (2 layers, 128 units); FC 128→64→1, Sigmoid | Linear 12→256; 6 residual blocks (RMSNorm, bidirectional Mamba-2 mixer, gated MLP); d_state = 64, d_conv = 4, expansion 2; attention pooling → 1 logit |
| Parameters | 367,169 | 10,708,770 |
| Loss | Weighted BCE (inverse-frequency class weights) | Focal loss (γ = 2, α = 0.75) |
| Optimizer | Adam; ReduceLROnPlateau (factor 0.5, patience 5) | AdamW, lr 3×10⁻⁴, weight decay 10⁻²; 5% warmup then cosine decay |
| Regularization | Dropout 0.3 | Dropout 0.2; gradient clipping (max norm 1.0) |
| Augmentation | — | Gaussian noise, amplitude scaling, time shift, baseline wander injection |

Within each architecture, hyperparameters were held constant across Datasets A, C and D. Evaluation metrics: accuracy, recall, precision, F1, specificity, AUROC, AUPRC, Brier score and Expected Calibration Error (ECE, 10 bins). Explainability uses Integrated Gradients for both architectures, with subclass-stratified analysis of anterior (AMI) and inferior (IMI) MI.

## Results and Analysis

All models were evaluated on the same certain-only test set (971 ECGs: 703 normal, 268 certain MI).

| Metric | CNN-LSTM A | CNN-LSTM C | CNN-LSTM D | Mamba-2 A | Mamba-2 C | Mamba-2 D |
|---|---|---|---|---|---|---|
| Training set (n) | 5,645 | 6,838 | 5,645 | 5,645 | 6,838 | 5,645 |
| Accuracy (%) | **96.81** | 95.57 | 93.72 | 95.37 | 96.29 | 92.17 |
| Recall (%) | 94.03 | 94.78 | 95.15 | 89.93 | **96.27** | **96.27** |
| Precision (%) | **94.38** | 89.75 | 84.16 | 93.05 | 90.85 | 79.63 |
| F1-score (%) | **94.21** | 92.20 | 89.32 | 91.46 | 93.48 | 87.16 |
| Specificity (%) | **97.87** | 95.87 | 93.17 | 97.44 | 96.30 | 90.61 |
| AUROC | **0.9933** | 0.9928 | 0.9891 | 0.9907 | 0.9919 | 0.9871 |
| AUPRC | **0.9849** | 0.9834 | 0.9754 | 0.9794 | 0.9814 | 0.9728 |
| Brier score (↓) | **0.0274** | 0.0326 | 0.0434 | 0.0341 | 0.0345 | 0.0647 |
| ECE (↓) | **0.0184** | 0.0233 | 0.0465 | 0.0339 | 0.0617 | 0.1377 |
| False positives (of 703) | **15** | 29 | 48 | 18 | 26 | 66 |
| Missed MI (of 268) | 16 | 14 | 13 | 27 | **10** | **10** |

Bold = best value across all six configurations.

**Key findings**

- **Certainty improves calibration.** Dataset A gives the lowest ECE and Brier score for both architectures. ECE rises from 0.0184 (A) to 0.0233 (C) and 0.0465 (D) for CNN-LSTM, and from 0.0339 (A) to 0.0617 (C) and 0.1377 (D) for Mamba-2.
- **Replacing certain labels with uncertain ones increases false positives.** Dataset D triggers the most false alarms (48 for CNN-LSTM, 66 for Mamba-2).
- **All-MI training improves sensitivity, especially for Mamba-2** (recall 96.27%, AUPRC 0.9814), while adding relatively few false positives compared with A.
- **Deployment guidance.** Dataset A suits probability-based risk stratification; Dataset C suits high-sensitivity screening; Dataset D is consistently unfavourable.

**Explainability (top-3 leads by Integrated Gradients importance)**

| Model | Dataset | AMI top 3 | IMI top 3 |
|---|---|---|---|
| CNN-LSTM | A | V2, V6, V3 | III, V2, V6 |
| CNN-LSTM | C | V2, V4, V3 | V2, aVF, III |
| CNN-LSTM | D | V3, V6, V4 | V2, V1, V6 |
| Bi-Mamba-2 | A | V1, V2, V3 | V2, V1, III |
| Bi-Mamba-2 | C | V1, V2, V3 | V1, III, V2 |
| Bi-Mamba-2 | D | V2, V1, V3 | V2, V1, aVL |
| *Expected (textbook)* | | V1, V2, V3 | II, III, aVF |

Anterior MI patterns are robust across all configurations. For inferior MI, Datasets A and C keep at least one inferior lead (III or aVF) in the top 3, while Dataset D does not, suggesting that uncertain training labels reduce the lead-territory alignment of learned features.

## Conclusion

Training exclusively on diagnostically certain MI cases produced the best-calibrated models across both architectures (CNN-LSTM ECE 0.0184; Mamba-2 ECE 0.0339). Including all MI cases improved sensitivity but degraded calibration, and the size-matched mixed cohort was consistently detrimental. The calibration ranking A > C > D held for both CNN-LSTM and Mamba-2, indicating that label certainty is an architecture-agnostic determinant of calibration quality: **label quality, not quantity, is the primary driver of probability trustworthiness in ECG-based MI detection.** We recommend that ECG AI studies routinely report calibration metrics (ECE, Brier score) alongside discrimination metrics.

**Limitations and future work:** evaluation on PTB-XL only (external validation needed), binary MI vs. normal rather than multi-label classification, different loss functions per architecture, and no post-hoc calibration analysis. Future work includes multi-label extension, soft-label uncertainty modeling, temperature scaling, loss-function ablations, label curation and synthetic data for scarce high-confidence cases.

## Publications

1. M. De Croos, N. Padeniya, C. Rupasinghe, I. Devindi, M. M. Maleckar, R. Ragel, J. K. Kanters, I. Nawinne and V. Thambawita, "Quality Over Quantity: The Impact of Diagnostic Certainty of Data in Deep Learning for ECG Analysis," *IEEE International Symposium on Computer-Based Medical Systems (CBMS)*, 2026. [Paper](https://www.computer.org/csdl/proceedings-article/cbms/2026/499900a068/2krxTvs42go)

*Acknowledgement:* This work is part of the European project SEARCH, supported by the Innovative Health Initiative Joint Undertaking (IHI JU) under grant agreement No. 101172997.

## Links

- [Published Paper (IEEE CBMS 2026)](https://www.computer.org/csdl/proceedings-article/cbms/2026/499900a068/2krxTvs42go)
- [Project Repository](https://github.com/cepdnaclk/e20-4yp-ECG-Based-MI-Detection)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-ECG-Based-MI-Detection)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

