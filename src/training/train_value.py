"""
Train Value LSTM model for time-series regression.
- Normalize with z-score (train stats)
- Compute μ+σ threshold on train normal errors
- Early stopping patience=5
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.dataset_loader import (
    load_event_traces, split_by_blockid, build_value_dataloaders
)
from src.models.value_lstm import ValueLSTM, compute_mse


def train_value_model(data_path='Event_traces.csv', h=10, batch_size=128, epochs=20, lr=1e-3, patience=5):
    """Train Value LSTM model."""
    # Setup
    set_seed(1337)
    device = get_device(prefer_mps=True, verbose=True)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts/training', exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_event_traces(data_path)
    train_df, val_df, test_df = split_by_blockid(df, seed=1337)
    
    # Build dataloaders with normalization
    print("\n[2/6] Building dataloaders with normalization...")
    train_loader, val_loader, test_loader, train_stats, metadata = build_value_dataloaders(
        train_df, val_df, test_df, h=h, batch_size=batch_size, normalize=True
    )
    
    # Save normalization stats
    with open('artifacts/training/value_norm.json', 'w') as f:
        json.dump(train_stats, f, indent=2)
    print(f"✓ Saved normalization stats: mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f}")
    
    # Create model
    print("\n[3/6] Creating model...")
    model = ValueLSTM(input_dim=1, hidden=64, num_layers=1, dropout=0.0)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\n[4/6] Training for {epochs} epochs...")
    history = {'train_mse': [], 'val_mse': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_mse = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_mse += loss.item()
        
        train_mse /= len(train_loader)
        
        # Validate
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_mse += loss.item()
        
        val_mse /= len(val_loader)
        
        # Record history
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        print(f"Epoch {epoch+1:02d}: Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}")
        
        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            model.save('models/deeplog_value_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save history
    with open('artifacts/training/value_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to artifacts/training/value_history.json")
    
    # Plot loss curves
    print("\n[5/6] Plotting loss curves...")
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Value LSTM: MSE Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/training/value_loss_curve.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss curves to artifacts/training/value_loss_curve.png")
    
    # Compute threshold on train normal samples
    print("\n[6/6] Computing anomaly threshold on train normal samples...")
    model = ValueLSTM.load('models/deeplog_value_model.pt', device=device)
    model.eval()
    
    # Get train normal errors
    train_normal_df = train_df[train_df['label_binary'] == 0]
    from src.data.dataset_loader import make_value_windows, ValueDataset
    from torch.utils.data import DataLoader
    
    X_train_normal, y_train_normal, _, _ = make_value_windows(
        train_normal_df, h=h, normalize=True, train_stats=train_stats
    )
    train_normal_loader = DataLoader(
        ValueDataset(X_train_normal, y_train_normal), batch_size=batch_size, shuffle=False
    )
    
    errors = []
    with torch.no_grad():
        for X_batch, y_batch in train_normal_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            error = ((pred - y_batch) ** 2).mean(dim=1).cpu().numpy()  # per-sample MSE
            errors.extend(error.tolist())
    
    errors = np.array(errors)
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    
    threshold_data = {
        'mean': mean_error,
        'std': std_error,
        'threshold_k2': mean_error + 2 * std_error,
        'threshold_k3': mean_error + 3 * std_error,
        'num_samples': len(errors)
    }
    
    with open('artifacts/training/value_threshold.json', 'w') as f:
        json.dump(threshold_data, f, indent=2)
    
    print(f"✓ Computed threshold statistics:")
    print(f"  Mean error (μ): {mean_error:.6f}")
    print(f"  Std error (σ): {std_error:.6f}")
    print(f"  Threshold (μ+2σ): {threshold_data['threshold_k2']:.6f}")
    print(f"  Threshold (μ+3σ): {threshold_data['threshold_k3']:.6f}")
    
    return model, history, threshold_data


if __name__ == '__main__':
    train_value_model()
    print("\n✓ Task Done: Value LSTM training completed")
    print("  - models/deeplog_value_model.pt")
    print("  - artifacts/training/value_history.json")
    print("  - artifacts/training/value_loss_curve.png")
    print("  - artifacts/training/value_norm.json")
    print("  - artifacts/training/value_threshold.json")
