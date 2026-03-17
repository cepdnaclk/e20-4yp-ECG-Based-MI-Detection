"""
Dataset B Evaluation: CNN-LSTM Test Set Performance
Evaluates on test set with breakdown by group (certain_mi vs uncertain_mi)
Training was on: pure_normal + uncertain_mi
"""

import sys
sys.path.append('../../shared_utils')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import model
from model import CNNLSTM

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# DATASET CLASS
# ==========================================

class ECGDataset(Dataset):
    """PyTorch Dataset for ECG signals"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# EVALUATION FUNCTIONS
# ==========================================

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    Returns predictions, probabilities, and true labels
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Get predictions
            outputs = model(batch_X)
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    return all_preds, all_probs, all_labels

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'specificity': None,
        'npv': None,
        'ppv': None
    }
    
    # Compute confusion matrix for additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return metrics

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'MI'],
                yticklabels=['Normal', 'MI'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_probability_distribution(y_true, y_prob, save_path):
    """Plot probability distribution by class"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Split by true label
    normal_probs = y_prob[y_true == 0]
    mi_probs = y_prob[y_true == 1]
    
    # Normal class
    axes[0].hist(normal_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_title('Probability Distribution - Normal Cases', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Probability (MI)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MI class
    axes[1].hist(mi_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_title('Probability Distribution - MI Cases', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Probability (MI)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_group_comparison(group_metrics, save_path):
    """Plot performance comparison across MI groups"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Get MI groups only (exclude pure_normal)
    mi_groups = [g for g in group_metrics.keys() if g != 'pure_normal']
    
    if len(mi_groups) == 0:
        print("⚠️ No MI groups to compare")
        return
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 3, idx % 3]
        
        values = [group_metrics[g][metric_key] for g in mi_groups if metric_key in group_metrics[g]]
        
        if len(values) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            continue
        
        bars = ax.bar(mi_groups, values, color=colors[:len(mi_groups)], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{metric_name} (vs Pure Normal)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

# ==========================================
# MAIN EVALUATION
# ==========================================

def main():
    print(f"\n{'='*60}")
    print("CNN-LSTM EVALUATION - DATASET B")
    print(f"{'='*60}")
    
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    
    print("\n[Step 1] Loading test data...")
    
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    groups_test = np.load('../data/groups_test.npy', allow_pickle=True)
    
    print(f"✅ Test data loaded:")
    print(f"   Shape: {X_test.shape}")
    print(f"   Labels: {y_test.shape}")
    print(f"   Total samples: {len(y_test)}")
    
    print(f"\nTest set composition:")
    unique_groups, group_counts = np.unique(groups_test, return_counts=True)
    for group, count in zip(unique_groups, group_counts):
        print(f"   {group}: {count} samples ({count/len(groups_test)*100:.2f}%)")
    
    print(f"\nOverall label distribution:")
    print(f"   Normal (0): {np.sum(y_test==0)} ({np.mean(y_test==0)*100:.2f}%)")
    print(f"   MI (1): {np.sum(y_test==1)} ({np.mean(y_test==1)*100:.2f}%)")
    
    # ==========================================
    # 2. LOAD MODEL
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 2] Loading trained model...")
    
    model = CNNLSTM(input_channels=12, lstm_hidden=128, dropout=0.3).to(device)
    
    # Load best model checkpoint
    checkpoint = torch.load('../saved_models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded:")
    print(f"   Checkpoint epoch: {checkpoint['epoch']}")
    print(f"   Validation AUC: {checkpoint['val_auc']:.4f}")
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # ==========================================
    # 3. EVALUATE ON TEST SET
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 3] Evaluating on test set...")
    
    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    y_pred, y_prob, y_true = evaluate_model(model, test_loader, device)
    
    print(f"✅ Predictions generated")
    
    # ==========================================
    # 4. COMPUTE OVERALL METRICS
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 4] Computing overall metrics...")
    
    overall_metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print(f"\n{'='*60}")
    print("OVERALL TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy:    {overall_metrics['accuracy']:.4f}")
    print(f"Precision:   {overall_metrics['precision']:.4f}")
    print(f"Recall:      {overall_metrics['recall']:.4f}")
    print(f"F1-Score:    {overall_metrics['f1']:.4f}")
    print(f"ROC-AUC:     {overall_metrics['roc_auc']:.4f}")
    print(f"Specificity: {overall_metrics['specificity']:.4f}")
    print(f"PPV:         {overall_metrics['ppv']:.4f}")
    print(f"NPV:         {overall_metrics['npv']:.4f}")
    print(f"{'='*60}")
    
    # ==========================================
    # 5. GROUP-WISE ANALYSIS
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 5] Computing group-wise metrics...")
    
    group_metrics = {}
    
    # For each MI group, compute metrics against pure_normal
    # This allows proper AUC calculation (requires both classes)
    
    for group in unique_groups:
        if group == 'pure_normal':
            continue  # Skip normal group for now
        
        # Create binary problem: current MI group vs pure_normal
        mi_mask = groups_test == group
        normal_mask = groups_test == 'pure_normal'
        combined_mask = mi_mask | normal_mask
        
        if np.sum(combined_mask) > 0:
            group_y_true = y_true[combined_mask]
            group_y_pred = y_pred[combined_mask]
            group_y_prob = y_prob[combined_mask]
            
            group_metrics[group] = compute_metrics(group_y_true, group_y_pred, group_y_prob)
            group_metrics[group]['n_samples'] = np.sum(mi_mask)
            group_metrics[group]['n_normal'] = np.sum(normal_mask)
    
    # Also compute metrics for pure_normal separately
    normal_mask = groups_test == 'pure_normal'
    if np.sum(normal_mask) > 0:
        normal_y_true = y_true[normal_mask]
        normal_y_pred = y_pred[normal_mask]
        
        # For normal group, just show accuracy/specificity
        tn = np.sum((normal_y_true == 0) & (normal_y_pred == 0))
        fp = np.sum((normal_y_true == 0) & (normal_y_pred == 1))
        
        group_metrics['pure_normal'] = {
            'accuracy': np.mean(normal_y_pred == normal_y_true),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'n_samples': np.sum(normal_mask),
            'true_negatives': tn,
            'false_positives': fp
        }
    
    print(f"\n{'='*60}")
    print("GROUP-WISE PERFORMANCE (vs Pure Normal)")
    print(f"{'='*60}")
    
    for group, metrics in group_metrics.items():
        if group == 'pure_normal':
            continue
        
        n_samples = metrics['n_samples']
        n_normal = metrics['n_normal']
        print(f"\n{group.upper()} vs PURE NORMAL:")
        print(f"  MI samples: {n_samples}, Normal samples: {n_normal}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
    
    # Pure normal performance
    if 'pure_normal' in group_metrics:
        metrics = group_metrics['pure_normal']
        print(f"\nPURE NORMAL:")
        print(f"  Total samples: {metrics['n_samples']}")
        print(f"  True Negatives: {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
    
    print(f"{'='*60}")
    
    # ==========================================
    # 6. GENERATE PLOTS
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 6] Generating visualizations...")
    
    save_dir = '../results/figures/'
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_dir + 'confusion_matrix_test.png')
    
    # ROC curve
    plot_roc_curve(y_true, y_prob, save_dir + 'roc_curve_test.png')
    
    # Probability distribution
    plot_probability_distribution(y_true, y_prob, save_dir + 'probability_distribution_test.png')
    
    # Group comparison
    plot_group_comparison(group_metrics, save_dir + 'group_comparison_test.png')
    
    # ==========================================
    # 7. SAVE RESULTS
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 7] Saving results...")
    
    # Save predictions
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'group': groups_test
    })
    results_df.to_csv('../results/metrics/test_predictions.csv', index=False)
    print(f"✅ Saved: test_predictions.csv")
    
    # Save overall metrics
    metrics_df = pd.DataFrame([overall_metrics])
    metrics_df.to_csv('../results/metrics/test_metrics.csv', index=False)
    print(f"✅ Saved: test_metrics.csv")
    
    # Save group-wise metrics
    group_metrics_df = pd.DataFrame(group_metrics).T
    group_metrics_df.to_csv('../results/metrics/test_group_metrics.csv')
    print(f"✅ Saved: test_group_metrics.csv")
    
    # Save detailed report
    with open('../results/metrics/evaluation_report.txt', 'w') as f:
        f.write("CNN-LSTM Evaluation Report - Dataset A\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL TEST SET PERFORMANCE\n")
        f.write("-"*60 + "\n")
        for key, value in overall_metrics.items():
            f.write(f"{key.upper():15s}: {value:.4f}\n")
        
        f.write("\n\nGROUP-WISE PERFORMANCE\n")
        f.write("-"*60 + "\n")
        for group, metrics in group_metrics.items():
            n_samples = np.sum(groups_test == group)
            f.write(f"\n{group.upper()} (n={n_samples}):\n")
            for key, value in metrics.items():
                f.write(f"  {key:15s}: {value:.4f}\n")
        
        f.write("\n\nCLASSIFICATION REPORT\n")
        f.write("-"*60 + "\n")
        f.write(classification_report(y_true, y_pred, 
                                     target_names=['Normal', 'MI']))
    
    print(f"✅ Saved: evaluation_report.txt")
    
    # ==========================================
    # 8. FINAL SUMMARY
    # ==========================================
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Overall Test Set Performance:")
    print(f"   Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC: {overall_metrics['roc_auc']:.4f}")
    print(f"   F1-Score: {overall_metrics['f1']:.4f}")
    print(f"   Recall: {overall_metrics['recall']:.4f}")
    print(f"   Precision: {overall_metrics['precision']:.4f}")
    
    print(f"\n🎯 Key Findings (MI Detection vs Pure Normal):")
    if 'certain_mi' in group_metrics and 'uncertain_mi' in group_metrics:
        certain_auc = group_metrics['certain_mi']['roc_auc']
        certain_recall = group_metrics['certain_mi']['recall']
        uncertain_auc = group_metrics['uncertain_mi']['roc_auc']
        uncertain_recall = group_metrics['uncertain_mi']['recall']
        
        print(f"\n   CERTAIN MI (trained on this):")
        print(f"      ROC-AUC: {certain_auc:.4f}")
        print(f"      Recall:  {certain_recall:.4f}")
        
        print(f"\n   UNCERTAIN MI (NOT trained on this):")
        print(f"      ROC-AUC: {uncertain_auc:.4f}")
        print(f"      Recall:  {uncertain_recall:.4f}")
        
        print(f"\n   📉 Performance Drop:")
        print(f"      AUC drop:    {(certain_auc - uncertain_auc)*100:.2f} percentage points")
        print(f"      Recall drop: {(certain_recall - uncertain_recall)*100:.2f} percentage points")
        
        print(f"\n   💡 Interpretation:")
        if uncertain_auc > 0.90:
            print(f"      Model generalizes WELL to uncertain MI cases!")
        elif uncertain_auc > 0.85:
            print(f"      Model shows GOOD generalization to uncertain MI cases.")
        else:
            print(f"      Model struggles with uncertain MI cases (as expected).")
    
    print(f"\n💾 All results saved to: ../results/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()