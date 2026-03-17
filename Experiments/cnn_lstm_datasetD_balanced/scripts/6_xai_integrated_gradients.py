"""
Explainability Analysis: Integrated Gradients with ECG Heatmap Visualization
Creates blue-to-red heatmap overlays on ECG signals showing model attention
Style matches: Nature Scientific Reports ECG XAI paper
"""

import sys
sys.path.append('../../shared_utils')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from captum.attr import IntegratedGradients
import warnings
warnings.filterwarnings('ignore')

from model import CNNLSTM

# ECG lead names
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = '../data/'
MODEL_PATH = '../saved_models/best_model.pth'
RESULTS_DIR = '../results/xai/'

# Create results directory
import os
os.makedirs(RESULTS_DIR + 'figures/', exist_ok=True)
os.makedirs(RESULTS_DIR + 'metrics/', exist_ok=True)

print(f"\n{'='*70}")
print("EXPLAINABILITY ANALYSIS - INTEGRATED GRADIENTS")
print("Heatmap Overlay Visualization")
print(f"{'='*70}")

# ==========================================
# 1. LOAD MODEL AND DATA
# ==========================================

print(f"\n[Step 1] Loading model and test data...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = CNNLSTM()
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"✅ Model loaded from: {MODEL_PATH}")

# Load test data
X_test = np.load(DATA_DIR + 'X_test.npy')
y_test = np.load(DATA_DIR + 'y_test.npy')
groups_test = np.load(DATA_DIR + 'groups_test.npy', allow_pickle=True)

print(f"✅ Test data loaded: {X_test.shape}")

# ==========================================
# 2. SELECT REPRESENTATIVE SAMPLES
# ==========================================

print(f"\n[Step 2] Selecting representative ECG samples...")

# Get predictions
X_test_tensor = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    y_pred_proba = model(X_test_tensor).cpu().numpy()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Select examples
mi_correct_idx = np.where((y_test == 1) & (y_pred == 1) & (groups_test == 'certain_mi'))[0]
mi_missed_idx = np.where((y_test == 1) & (y_pred == 0))[0]
normal_correct_idx = np.where((y_test == 0) & (y_pred == 0))[0]
false_positive_idx = np.where((y_test == 0) & (y_pred == 1))[0]

examples = {
    'MI_Correct_High_Conf': mi_correct_idx[np.argmax(y_pred_proba[mi_correct_idx])] if len(mi_correct_idx) > 0 else None,
    'MI_Correct_Medium_Conf': mi_correct_idx[len(mi_correct_idx)//2] if len(mi_correct_idx) > 0 else None,
    'MI_Missed': mi_missed_idx[0] if len(mi_missed_idx) > 0 else None,
    'Normal_Correct': normal_correct_idx[0] if len(normal_correct_idx) > 0 else None,
    'False_Positive': false_positive_idx[0] if len(false_positive_idx) > 0 else None,
}

print(f"\n📊 Selected Examples:")
for name, idx in examples.items():
    if idx is not None:
        prob = y_pred_proba[idx]
        pred = "MI" if y_pred[idx] == 1 else "Normal"
        truth = "MI" if y_test[idx] == 1 else "Normal"
        print(f"  {name}: Pred={pred} ({prob:.1%}), Truth={truth}")

# ==========================================
# 3. INTEGRATED GRADIENTS COMPUTATION
# ==========================================

print(f"\n[Step 3] Computing Integrated Gradients attributions...")

# Initialize Integrated Gradients
# Wrapper to enable train mode for backward pass but keep dropout/batchnorm in eval mode
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Temporarily enable train mode for LSTM gradient computation
        self.model.train()
        # Disable dropout/batchnorm effects
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d)):
                module.eval()
        output = self.model(x)
        self.model.eval()
        return output

model_wrapper = ModelWrapper(model)
ig = IntegratedGradients(model_wrapper)

def compute_attributions(ecg_sample):
    """
    Compute Integrated Gradients attributions
    
    Args:
        ecg_sample: (1000, 12) ECG signal
    
    Returns:
        attributions: (1000, 12) importance scores
    """
    input_tensor = torch.FloatTensor(ecg_sample).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # Baseline: zeros (no signal)
    baseline = torch.zeros_like(input_tensor).to(device)
    
    # Compute attributions (no target needed for binary classification with single output)
    attributions = ig.attribute(input_tensor, baselines=baseline, n_steps=50)
    
    return attributions.squeeze().cpu().detach().numpy()

# Compute attributions for all examples
attributions_dict = {}
for name, idx in examples.items():
    if idx is not None:
        print(f"  Computing: {name}...")
        attrs = compute_attributions(X_test[idx])
        attributions_dict[name] = attrs

print(f"✅ Attributions computed for {len(attributions_dict)} examples")

# ==========================================
# 4. CREATE HEATMAP OVERLAY VISUALIZATIONS
# ==========================================

print(f"\n[Step 4] Creating heatmap overlay visualizations...")

# Custom colormap: Blue (low importance) -> Yellow -> Red (high importance)
colors = ['#0000FF', '#4040FF', '#8080FF', '#FFFFFF', '#FFFF80', '#FFAA00', '#FF0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('ecg_importance', colors, N=n_bins)

def plot_ecg_with_heatmap(ecg_signal, attributions, lead_idx, ax, title=""):
    """
    Plot single ECG lead with heatmap background
    
    Args:
        ecg_signal: (1000,) ECG signal for one lead
        attributions: (1000,) attribution scores for one lead
        lead_idx: Lead index (0-11)
        ax: Matplotlib axis
        title: Plot title
    """
    timesteps = np.arange(len(ecg_signal))
    
    # Normalize attributions to [0, 1]
    attr_norm = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
    
    # Create heatmap background using pcolormesh
    # Create a 2D array for heatmap (repeat attribution values vertically)
    heatmap_data = np.tile(attr_norm, (100, 1))
    
    # Plot heatmap
    extent = [0, len(ecg_signal), ecg_signal.min() - 0.5, ecg_signal.max() + 0.5]
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, extent=extent, 
                   origin='lower', alpha=0.7, interpolation='bilinear')
    
    # Plot ECG signal on top
    ax.plot(timesteps, ecg_signal, 'k-', linewidth=1.5, zorder=10)
    
    # Formatting
    ax.set_xlim(0, len(ecg_signal))
    ax.set_ylim(ecg_signal.min() - 0.3, ecg_signal.max() + 0.3)
    ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Remove x-axis labels except for bottom plot
    if lead_idx < 11:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (samples)', fontsize=10)
    
    return im

def create_full_12lead_heatmap(ecg_sample, attributions, sample_name, pred_prob, true_label):
    """
    Create 12-lead ECG visualization with heatmap overlays
    
    Args:
        ecg_sample: (1000, 12) ECG signal
        attributions: (1000, 12) attribution scores
        sample_name: Name for the figure
        pred_prob: Predicted probability
        true_label: True label (0=Normal, 1=MI)
    """
    fig, axes = plt.subplots(12, 1, figsize=(14, 16))
    fig.suptitle(f'{sample_name}\nPredicted MI Probability: {pred_prob:.1%}, True: {"MI" if true_label==1 else "Normal"}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Plot each lead
    for lead_idx in range(12):
        im = plot_ecg_with_heatmap(
            ecg_sample[:, lead_idx], 
            attributions[:, lead_idx],
            lead_idx, 
            axes[lead_idx],
            title=f"Lead {LEAD_NAMES[lead_idx]}"
        )
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Model Attention\n(Importance)', fontsize=11, fontweight='bold')
    cbar.ax.set_yticklabels(['Low', '', '', '', 'High'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    
    # Save
    filename = f"{sample_name.replace(' ', '_')}_12lead_heatmap.png"
    plt.savefig(RESULTS_DIR + 'figures/' + filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def create_selected_leads_heatmap(ecg_sample, attributions, sample_name, pred_prob, true_label, leads=[1, 5, 7, 10]):
    """
    Create visualization with selected clinically important leads
    (Lead II, aVF, V2, V5)
    
    Args:
        ecg_sample: (1000, 12) ECG signal
        attributions: (1000, 12) attribution scores
        sample_name: Name for the figure
        pred_prob: Predicted probability
        true_label: True label
        leads: List of lead indices to plot
    """
    fig, axes = plt.subplots(len(leads), 1, figsize=(14, 8))
    fig.suptitle(f'{sample_name}\nPredicted: {pred_prob:.1%} MI, True: {"MI" if true_label==1 else "Normal"}',
                 fontsize=13, fontweight='bold')
    
    for i, lead_idx in enumerate(leads):
        ax = axes[i] if len(leads) > 1 else axes
        im = plot_ecg_with_heatmap(
            ecg_sample[:, lead_idx],
            attributions[:, lead_idx],
            lead_idx,
            ax,
            title=f"Lead {LEAD_NAMES[lead_idx]}"
        )
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Importance', fontsize=11, fontweight='bold')
    cbar.ax.set_yticklabels(['Low', '', 'Medium', '', 'High'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    
    filename = f"{sample_name.replace(' ', '_')}_selected_leads.png"
    plt.savefig(RESULTS_DIR + 'figures/' + filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

# Generate visualizations for each example
for name, idx in examples.items():
    if idx is not None and name in attributions_dict:
        ecg = X_test[idx]
        attrs = attributions_dict[name]
        prob = y_pred_proba[idx]
        label = y_test[idx]
        
        # Full 12-lead visualization
        create_full_12lead_heatmap(ecg, attrs, name, prob, label)
        
        # Selected leads (II, aVF, V2, V5) - clinically important for MI
        create_selected_leads_heatmap(ecg, attrs, name, prob, label, leads=[1, 5, 7, 10])

# ==========================================
# 5. LEAD-WISE IMPORTANCE ANALYSIS
# ==========================================

print(f"\n[Step 5] Analyzing lead-wise importance...")

# Compute average importance for each lead across MI cases
mi_indices = np.where((y_test == 1) & (y_pred == 1))[0][:30]  # First 30 correctly classified MI
lead_importances = []

for idx in mi_indices:
    attrs = compute_attributions(X_test[idx])
    # Sum absolute attributions over time for each lead
    lead_imp = np.abs(attrs).sum(axis=0)
    lead_importances.append(lead_imp)

lead_importances = np.array(lead_importances)
avg_lead_importance = lead_importances.mean(axis=0)
std_lead_importance = lead_importances.std(axis=0)

# Normalize
avg_lead_importance_norm = avg_lead_importance / avg_lead_importance.sum()

# Plot lead importance
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(12)
ax.bar(x, avg_lead_importance_norm, yerr=std_lead_importance/avg_lead_importance.sum(), 
       color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(LEAD_NAMES, fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
ax.set_xlabel('ECG Lead', fontsize=12, fontweight='bold')
ax.set_title('Lead-Wise Importance for MI Detection\n(Averaged over 30 MI Cases)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(avg_lead_importance_norm) * 1.2)

# Add importance values on top of bars
for i, (imp, std) in enumerate(zip(avg_lead_importance_norm, std_lead_importance/avg_lead_importance.sum())):
    ax.text(i, imp + std + 0.005, f'{imp:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/lead_importance_bar_chart.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: lead_importance_bar_chart.png")
plt.close()

# Print importance table
print(f"\n📊 Lead Importance Rankings:")
importance_df = pd.DataFrame({
    'Lead': LEAD_NAMES,
    'Importance': avg_lead_importance_norm,
    'Std': std_lead_importance / avg_lead_importance.sum()
})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))

# Save to CSV
importance_df.to_csv(RESULTS_DIR + 'metrics/lead_importance.csv', index=False)
print(f"\n✅ Saved: lead_importance.csv")

# ==========================================
# 6. TEMPORAL IMPORTANCE ANALYSIS
# ==========================================

print(f"\n[Step 6] Analyzing temporal importance patterns...")

# Average attribution over all leads for each timestep
temporal_importance = np.abs(attributions_dict['MI_Correct_High_Conf']).mean(axis=1)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(temporal_importance, linewidth=2, color='darkred')
ax.fill_between(range(len(temporal_importance)), temporal_importance, alpha=0.3, color='red')
ax.set_xlabel('Time (samples @ 100 Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Importance', fontsize=12, fontweight='bold')
ax.set_title('Temporal Importance Pattern (Averaged Across All Leads)\nMI Case Example', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Mark important regions
max_importance_idx = np.argmax(temporal_importance)
ax.axvline(max_importance_idx, color='red', linestyle='--', linewidth=2, label=f'Peak at sample {max_importance_idx}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'figures/temporal_importance_profile.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: temporal_importance_profile.png")
plt.close()

# ==========================================
# 7. SUMMARY REPORT
# ==========================================

print(f"\n{'='*70}")
print("EXPLAINABILITY ANALYSIS COMPLETE!")
print(f"{'='*70}")

print(f"\n📁 Generated Visualizations:")
print(f"  • 12-lead heatmap overlays for each example")
print(f"  • Selected leads (II, aVF, V2, V5) visualizations")
print(f"  • Lead-wise importance bar chart")
print(f"  • Temporal importance profile")

print(f"\n📊 Key Findings:")
top_3_leads = importance_df.head(3)['Lead'].values
print(f"  • Top 3 most important leads: {', '.join(top_3_leads)}")
print(f"  • Most important lead: {top_3_leads[0]} ({importance_df.iloc[0]['Importance']:.1%})")
print(f"  • Peak temporal importance at sample {max_importance_idx} (~{max_importance_idx/100:.2f} seconds)")

print(f"\n💾 All results saved in: {RESULTS_DIR}")
print(f"{'='*70}\n")