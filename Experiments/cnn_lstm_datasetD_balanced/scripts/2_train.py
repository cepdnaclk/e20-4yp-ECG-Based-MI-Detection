"""
Dataset A Training: CNN-LSTM for Normal vs Certain MI
Training on pure_normal + certain_mi (MI with prob=100%)
"""

import sys
sys.path.append('../../shared_utils')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import model from shared_utils
from model import CNNLSTM

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
# TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, train_loader, criterion, optimizer, device, class_weights):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        
        # Compute weighted loss
        loss = criterion(outputs, batch_y)
        weights = torch.where(batch_y == 1, class_weights[1], class_weights[0])
        loss = (loss * weights).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
        
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, accuracy, auc

def validate(model, val_loader, criterion, device, class_weights):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            
            # Compute weighted loss
            loss = criterion(outputs, batch_y)
            weights = torch.where(batch_y == 1, class_weights[1], class_weights[0])
            loss = (loss * weights).mean()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, accuracy, auc

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_training_history(history, save_dir='../results/figures/'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    axes[2].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    axes[2].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('ROC-AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir + 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}training_history.png")
    plt.close()

# ==========================================
# MAIN TRAINING
# ==========================================

def main():
    print(f"\n{'='*60}")
    print("CNN-LSTM TRAINING - DATASET D (Balanced MI)")
    print(f"{'='*60}")
    
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    
    print("\n[Step 1] Loading preprocessed data...")
    
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')
    
    print(f"✅ Data loaded:")
    print(f"   Train: {X_train.shape} - {y_train.shape}")
    print(f"   Val: {X_val.shape} - {y_val.shape}")
    
    print(f"\nClass distribution:")
    print(f"   Train - Normal: {np.sum(y_train==0)} ({np.mean(y_train==0)*100:.2f}%)")
    print(f"          MI: {np.sum(y_train==1)} ({np.mean(y_train==1)*100:.2f}%)")
    print(f"   Val   - Normal: {np.sum(y_val==0)} ({np.mean(y_val==0)*100:.2f}%)")
    print(f"          MI: {np.sum(y_val==1)} ({np.mean(y_val==1)*100:.2f}%)")
    
    # ==========================================
    # 2. CREATE DATALOADERS
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 2] Creating DataLoaders...")
    
    # Class weights (from preprocessing)
    n_normal = np.sum(y_train == 0)
    n_mi = np.sum(y_train == 1)
    total = len(y_train)
    weight_normal = total / (2 * n_normal)
    weight_mi = total / (2 * n_mi)
    class_weights = torch.FloatTensor([weight_normal, weight_mi]).to(device)
    
    print(f"✅ Class weights: Normal={weight_normal:.3f}, MI={weight_mi:.3f}")
    
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # ==========================================
    # 3. INITIALIZE MODEL
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 3] Initializing CNN-LSTM model...")
    
    model = CNNLSTM(input_channels=12, lstm_hidden=128, dropout=0.3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model initialized:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ==========================================
    # 4. SETUP TRAINING
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 4] Setting up training configuration...")
    
    criterion = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training settings
    num_epochs = 100
    patience = 15
    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    
    # History
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }
    
    print(f"✅ Training configuration:")
    print(f"   Max epochs: {num_epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Learning rate: 0.001")
    print(f"   Batch size: 32")
    print(f"   Optimizer: Adam")
    print(f"   LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # ==========================================
    # 5. TRAIN MODEL
    # ==========================================
    
    print(f"\n{'='*60}")
    print("TRAINING START")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        
        # Validate
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device, class_weights
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Time: {epoch_time:5.1f}s | "
              f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
              f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model (based on validation AUC)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_acc': val_acc
            }
            torch.save(checkpoint, '../saved_models/best_model.pth')
            print(f"  ✅ Best model saved! (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
    
    total_time = time.time() - start_time
    
    # ==========================================
    # 6. SAVE RESULTS
    # ==========================================
    
    print(f"\n{'='*60}")
    print("[Step 6] Saving results...")
    
    # Save final model
    final_checkpoint = {
        'epoch': len(history['train_loss']),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': history['val_loss'][-1],
        'val_auc': history['val_auc'][-1],
        'val_acc': history['val_acc'][-1]
    }
    torch.save(final_checkpoint, '../saved_models/final_model.pth')
    
    # Save training history
    np.save('../results/metrics/training_history.npy', history)
    
    # Plot training curves
    plot_training_history(history)
    
    # Save training summary
    with open('../results/metrics/training_summary.txt', 'w') as f:
        f.write("CNN-LSTM Training Summary - Dataset A\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
        f.write(f"Total Epochs: {len(history['train_loss'])}\n\n")
        f.write(f"Best Validation Metrics:\n")
        f.write(f"  Loss: {best_val_loss:.4f}\n")
        f.write(f"  AUC: {best_val_auc:.4f}\n\n")
        f.write(f"Final Validation Metrics:\n")
        f.write(f"  Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Accuracy: {history['val_acc'][-1]:.4f}\n")
        f.write(f"  AUC: {history['val_auc'][-1]:.4f}\n\n")
        f.write(f"Model Parameters: {trainable_params:,}\n")
    
    print(f"✅ Training summary saved")
    
    # ==========================================
    # 7. FINAL SUMMARY
    # ==========================================
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Training Summary:")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Total epochs: {len(history['train_loss'])}")
    print(f"   Best validation AUC: {best_val_auc:.4f}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"\n💾 Saved files:")
    print(f"   • best_model.pth (best validation AUC)")
    print(f"   • final_model.pth (last epoch)")
    print(f"   • training_history.npy")
    print(f"   • training_history.png")
    print(f"   • training_summary.txt")
    print(f"\n▶️  Next step: Run 3_evaluate.py to evaluate on test set")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()