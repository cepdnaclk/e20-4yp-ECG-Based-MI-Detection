"""
Calibration Analysis: AUPRC and Calibration Plots
Analyzes prediction calibration across all three datasets (A, B, C)
"""

import sys
sys.path.append('../../shared_utils')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    brier_score_loss,
    roc_curve,
    auc
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def compute_ece(y_true, y_pred_proba, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def load_dataset_predictions(dataset_name):
    """
    Load predictions for a specific dataset
    
    Args:
        dataset_name: 'A', 'B', or 'C'
    
    Returns:
        y_true, y_pred, y_proba, groups
    """
    if dataset_name == 'A':
        path = '../cnn_lstm_datasetA_certain_mi/results/metrics/test_predictions.csv'
    elif dataset_name == 'B':
        path = '../cnn_lstm_datasetB_uncertain_mi/results/metrics/test_predictions.csv'
    elif dataset_name == 'C':
        path = '../cnn_lstm_datasetC_all_mi/results/metrics/test_predictions.csv'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    df = pd.read_csv(path)
    return df['y_true'].values, df['y_pred'].values, df['y_prob'].values, df['group'].values

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_precision_recall_curve(datasets_dict, save_path):
    """
    Plot Precision-Recall curves for all datasets
    
    Args:
        datasets_dict: Dict with dataset names as keys and (y_true, y_proba) as values
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}
    
    for dataset_name, (y_true, y_proba) in datasets_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, 
               label=f'Dataset {dataset_name} (AUPRC = {ap_score:.4f})',
               linewidth=2.5, color=colors[dataset_name])
    
    # Baseline (random classifier)
    baseline = np.sum(datasets_dict['A'][0]) / len(datasets_dict['A'][0])
    ax.plot([0, 1], [baseline, baseline], 
           'k--', label=f'Random Classifier (AP = {baseline:.4f})', linewidth=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves: Dataset Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_calibration_curve(datasets_dict, save_path, n_bins=10):
    """
    Plot calibration curves for all datasets
    
    Args:
        datasets_dict: Dict with dataset names as keys and (y_true, y_proba) as values
        save_path: Path to save figure
        n_bins: Number of bins for calibration
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}
    
    for dataset_name, (y_true, y_proba) in datasets_dict.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 
               marker='o', linewidth=2.5, markersize=8,
               label=f'Dataset {dataset_name}',
               color=colors[dataset_name])
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives (Actual MI Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curves: Dataset Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_reliability_diagram(datasets_dict, save_path, n_bins=10):
    """
    Plot reliability diagram (histogram + calibration)
    
    Args:
        datasets_dict: Dict with dataset names as keys and (y_true, y_proba) as values
        save_path: Path to save figure
        n_bins: Number of bins
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    colors = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}
    titles = {'A': 'Dataset A (Certain MI)', 'B': 'Dataset B (Uncertain MI)', 'C': 'Dataset C (All MI)'}
    
    for idx, (dataset_name, (y_true, y_proba)) in enumerate(datasets_dict.items()):
        # Calibration curve
        ax1 = axes[idx, 0]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', linewidth=2.5, markersize=10,
                color=colors[dataset_name], label='Model')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax1.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax1.set_ylabel('Fraction of Positives', fontsize=11)
        ax1.set_title(f'{titles[dataset_name]} - Calibration', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2 = axes[idx, 1]
        ax2.hist(y_proba[y_true == 0], bins=20, alpha=0.5, label='Normal', color='green', edgecolor='black')
        ax2.hist(y_proba[y_true == 1], bins=20, alpha=0.5, label='MI', color='red', edgecolor='black')
        ax2.set_xlabel('Predicted Probability', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title(f'{titles[dataset_name]} - Prediction Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_decile_calibration(datasets_dict, save_path):
    """
    Plot decile-based calibration (medical standard)
    
    Args:
        datasets_dict: Dict with dataset names as keys and (y_true, y_proba) as values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'A': '#e74c3c', 'B': '#2ecc71', 'C': '#3498db'}
    titles = {'A': 'Dataset A', 'B': 'Dataset B', 'C': 'Dataset C'}
    
    for idx, (dataset_name, (y_true, y_proba)) in enumerate(datasets_dict.items()):
        ax = axes[idx]
        
        # Create deciles
        deciles = pd.qcut(y_proba, q=10, duplicates='drop', labels=False)
        
        decile_stats = []
        for decile in range(deciles.max() + 1):
            mask = deciles == decile
            if mask.sum() > 0:
                mean_pred = y_proba[mask].mean()
                mean_actual = y_true[mask].mean()
                count = mask.sum()
                decile_stats.append({
                    'decile': decile + 1,
                    'mean_pred': mean_pred,
                    'mean_actual': mean_actual,
                    'count': count
                })
        
        decile_df = pd.DataFrame(decile_stats)
        
        # Plot
        ax.scatter(decile_df['mean_pred'], decile_df['mean_actual'], 
                  s=decile_df['count']*2, alpha=0.6, color=colors[dataset_name], 
                  edgecolors='black', linewidths=1.5)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Add decile numbers
        for _, row in decile_df.iterrows():
            ax.annotate(f"{int(row['decile'])}", 
                       (row['mean_pred'], row['mean_actual']),
                       fontsize=9, fontweight='bold', ha='center', va='center')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Observed MI Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'{titles[dataset_name]}\nDecile-Based Calibration', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

# ==========================================
# MAIN ANALYSIS
# ==========================================

def main():
    print(f"\n{'='*70}")
    print("CALIBRATION ANALYSIS: AUPRC & CALIBRATION PLOTS")
    print(f"{'='*70}")
    
    # ==========================================
    # 1. LOAD ALL DATASETS
    # ==========================================
    
    print("\n[Step 1] Loading predictions from all datasets...")
    
    datasets = {}
    for dataset_name in ['A', 'B', 'C']:
        y_true, y_pred, y_proba, groups = load_dataset_predictions(dataset_name)
        datasets[dataset_name] = (y_true, y_proba)
        print(f"✅ Dataset {dataset_name}: {len(y_true)} samples")
    
    # ==========================================
    # 2. COMPUTE METRICS
    # ==========================================
    
    print(f"\n{'='*70}")
    print("[Step 2] Computing calibration metrics...")
    print(f"{'='*70}")
    
    results = []
    
    for dataset_name, (y_true, y_proba) in datasets.items():
        # AUPRC
        ap_score = average_precision_score(y_true, y_proba)
        
        # Brier Score (lower is better)
        brier = brier_score_loss(y_true, y_proba)
        
        # ECE (lower is better)
        ece = compute_ece(y_true, y_proba, n_bins=10)
        
        # ROC-AUC for reference
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_proba)
        
        results.append({
            'Dataset': dataset_name,
            'AUPRC': ap_score,
            'ROC-AUC': roc_auc,
            'Brier Score': brier,
            'ECE': ece
        })
        
        print(f"\nDataset {dataset_name}:")
        print(f"  AUPRC:       {ap_score:.4f}")
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  Brier Score: {brier:.4f} (lower is better)")
        print(f"  ECE:         {ece:.4f} (lower is better)")
    
    # ==========================================
    # 3. CREATE RESULTS TABLE
    # ==========================================
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('calibration_metrics_comparison.csv', index=False)
    print(f"\n✅ Saved: calibration_metrics_comparison.csv")
    
    # ==========================================
    # 4. GENERATE PLOTS
    # ==========================================
    
    print(f"\n{'='*70}")
    print("[Step 3] Generating visualizations...")
    print(f"{'='*70}\n")
    
    # Precision-Recall Curves
    plot_precision_recall_curve(datasets, 'precision_recall_curves_comparison.png')
    
    # Calibration Curves
    plot_calibration_curve(datasets, 'calibration_curves_comparison.png', n_bins=10)
    
    # Reliability Diagrams
    plot_reliability_diagram(datasets, 'reliability_diagrams.png', n_bins=10)
    
    # Decile Calibration
    plot_decile_calibration(datasets, 'decile_calibration_comparison.png')
    
    # ==========================================
    # 5. INTERPRETATION
    # ==========================================
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    # Find best dataset for each metric
    best_auprc = results_df.loc[results_df['AUPRC'].idxmax(), 'Dataset']
    best_brier = results_df.loc[results_df['Brier Score'].idxmin(), 'Dataset']
    best_ece = results_df.loc[results_df['ECE'].idxmin(), 'Dataset']
    
    print(f"\n🏆 Best Performance:")
    print(f"  Best AUPRC (Precision-Recall):  Dataset {best_auprc}")
    print(f"  Best Brier Score (Calibration): Dataset {best_brier}")
    print(f"  Best ECE (Calibration):         Dataset {best_ece}")
    
    print(f"\n💡 Calibration Insights:")
    print(f"  - AUPRC measures overall precision-recall trade-off")
    print(f"  - Brier Score measures probability accuracy")
    print(f"  - ECE measures calibration error across probability bins")
    print(f"  - Lower Brier & ECE = Better calibrated probabilities")
    
    print(f"\n📊 Clinical Relevance:")
    print(f"  - Well-calibrated model → Clinicians can trust probabilities")
    print(f"  - If model says '80% MI risk', ~80% should actually have MI")
    print(f"  - Poor calibration → Model overconfident or underconfident")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Files Generated:")
    print(f"  • calibration_metrics_comparison.csv")
    print(f"  • precision_recall_curves_comparison.png")
    print(f"  • calibration_curves_comparison.png")
    print(f"  • reliability_diagrams.png")
    print(f"  • decile_calibration_comparison.png")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()