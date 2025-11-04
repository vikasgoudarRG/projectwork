"""
Train Key LSTM model for next-event classification.
- Window size h=10
- Top-g recall with g=9
- Early stopping patience=5
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.dataset_loader import (
    load_event_traces, split_by_blockid, build_key_dataloaders, get_vocab_size
)
from src.models.key_lstm import KeyLSTM, top_g_recall, compute_accuracy


def train_key_model(data_path='Event_traces.csv', h=10, batch_size=128, epochs=20, lr=1e-3, patience=5):
    """Train Key LSTM model."""
    # Setup
    set_seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts/training', exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_event_traces(data_path)
    train_df, val_df, test_df = split_by_blockid(df, seed=1337)
    
    # Build dataloaders
    print("\n[2/5] Building dataloaders...")
    train_loader, val_loader, test_loader, metadata = build_key_dataloaders(
        train_df, val_df, test_df, h=h, batch_size=batch_size
    )
    
    # Get vocab size
    vocab_size = get_vocab_size()
    print(f"  Vocab size: {vocab_size}")
    
    # Create model
    print("\n[3/5] Creating model...")
    model = KeyLSTM(vocab_size=vocab_size, embed_dim=64, hidden=64, num_layers=2, dropout=0.2)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\n[4/5] Training for {epochs} epochs...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_top9_recall': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_acc = 0, 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += compute_accuracy(logits, y_batch)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss, val_acc, val_topg = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                val_acc += compute_accuracy(logits, y_batch)
                val_topg += top_g_recall(logits, y_batch, g=9)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_topg /= len(val_loader)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_top9_recall'].append(val_topg)
        
        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, Val Top-9={val_topg:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save('models/deeplog_key_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save history
    with open('artifacts/training/key_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to artifacts/training/key_history.json")
    
    # Plot loss curves
    print("\n[5/5] Plotting loss curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Key LSTM: Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.plot(history['val_top9_recall'], label='Val Top-9 Recall')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy / Recall')
    ax2.set_title('Key LSTM: Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/training/key_loss_curve.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss curves to artifacts/training/key_loss_curve.png")
    
    # Final test evaluation
    print("\n[Test] Evaluating on test set...")
    model = KeyLSTM.load('models/deeplog_key_model.pt', device=device)
    model.eval()
    test_acc, test_topg = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            test_acc += compute_accuracy(logits, y_batch)
            test_topg += top_g_recall(logits, y_batch, g=9)
    
    test_acc /= len(test_loader)
    test_topg /= len(test_loader)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Top-9 Recall: {test_topg:.4f}")
    
    return model, history


if __name__ == '__main__':
    train_key_model()
    print("\n✓ Task Done: Key LSTM training completed")
    print("  - models/deeplog_key_model.pt")
    print("  - artifacts/training/key_history.json")
    print("  - artifacts/training/key_loss_curve.png")
