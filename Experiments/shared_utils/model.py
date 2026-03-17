"""
CNN-LSTM Model Architecture for ECG Classification
Hybrid model combining CNN for spatial features and LSTM for temporal patterns
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM Model for 12-lead ECG Classification
    
    Architecture:
        - 3 CNN blocks for spatial feature extraction
        - 1 LSTM layer for temporal modeling
        - Dense layers for classification
    
    Input: (batch_size, 1000, 12) - ECG signals
    Output: (batch_size, 1) - Probability of MI
    """
    
    def __init__(self, input_channels=12, lstm_hidden=128, dropout=0.3):
        super(CNNLSTM, self).__init__()
        
        # CNN Feature Extraction
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout + 0.1)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0  # Only 1 layer, so dropout=0
        )
        self.lstm_dropout = nn.Dropout(dropout + 0.1)
        
        # Dense classification layers
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc1_dropout = nn.Dropout(dropout + 0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc2_dropout = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, timesteps, channels)
        
        Returns:
            Output probability (batch_size, 1)
        """
        # Input: (batch, 1000, 12)
        # Conv1d expects: (batch, channels, timesteps)
        x = x.permute(0, 2, 1)  # (batch, 12, 1000)
        
        # CNN blocks
        x = self.conv_block1(x)  # (batch, 64, 500)
        x = self.conv_block2(x)  # (batch, 128, 250)
        x = self.conv_block3(x)  # (batch, 256, 125)
        
        # Prepare for LSTM: (batch, timesteps, features)
        x = x.permute(0, 2, 1)  # (batch, 125, 256)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        x = h_n[-1]  # (batch, lstm_hidden)
        x = self.lstm_dropout(x)
        
        # Dense layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc2_dropout(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x.squeeze()
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_summary(model, input_shape=(1, 1000, 12)):
    """
    Print model summary
    
    Args:
        model: CNN-LSTM model
        input_shape: Input tensor shape (batch, timesteps, channels)
    """
    print(f"\n{'='*60}")
    print("Model Architecture Summary")
    print(f"{'='*60}")
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print("✅ Model forward pass successful!")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
    
    print(f"{'='*60}\n")
    
    return total_params


def load_model_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model weights from checkpoint
    
    Args:
        model: CNN-LSTM model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        model: Model with loaded weights
        checkpoint: Full checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    if 'val_auc' in checkpoint:
        print(f"   Val AUC: {checkpoint['val_auc']:.4f}")
    
    return model, checkpoint


def save_model_checkpoint(model, optimizer, epoch, val_loss, val_auc, 
                          save_path, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: CNN-LSTM model
        optimizer: Optimizer
        epoch: Current epoch
        val_loss: Validation loss
        val_auc: Validation AUC
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_auc': val_auc,
        'is_best': is_best
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        print(f"✅ Best model saved! (AUC: {val_auc:.4f})")