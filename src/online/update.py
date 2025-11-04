"""
Online learning: Fine-tune Key LSTM on false positive samples.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.dataset_loader import (
    load_event_traces, split_by_blockid, make_key_windows, KeyDataset
)
from src.models.key_lstm import KeyLSTM
from torch.utils.data import DataLoader


def online_finetune(false_positive_blockids=None, epochs=2, lr=1e-4, batch_size=64):
    """
    Fine-tune Key LSTM on false positive samples.
    
    Args:
        false_positive_blockids: List of BlockIds that were false positives
        epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning
        batch_size: Batch size
    """
    # Setup
    set_seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('artifacts/online', exist_ok=True)
    
    # Load detection results to find false positives
    print("\n[1/4] Identifying false positives...")
    
    if false_positive_blockids is None:
        import pandas as pd
        session_df = pd.read_csv('artifacts/detection/session_anomalies.csv')
        
        # False positives: predicted=1, true=0
        fp_mask = (session_df['fused_anomaly'] == 1) & (session_df['true_label'] == 0)
        false_positive_blockids = session_df[fp_mask]['block_id'].tolist()
    
    print(f"  Found {len(false_positive_blockids)} false positive BlockIds")
    
    if len(false_positive_blockids) == 0:
        print("  No false positives to fine-tune on. Exiting.")
        return
    
    # Load data
    print("\n[2/4] Loading training data...")
    df = load_event_traces('Event_traces.csv')
    train_df, _, _ = split_by_blockid(df, seed=1337)
    
    # Filter to false positive blocks
    fp_df = train_df[train_df['BlockId'].isin(false_positive_blockids)]
    print(f"  Fine-tuning on {len(fp_df)} sequences")
    
    if len(fp_df) == 0:
        print("  No matching sequences in training set. Exiting.")
        return
    
    # Create windows
    X_fp, y_fp, _ = make_key_windows(fp_df, h=10)
    fp_loader = DataLoader(KeyDataset(X_fp, y_fp), batch_size=batch_size, shuffle=True)
    
    # Load model
    print("\n[3/4] Loading model for fine-tuning...")
    model = KeyLSTM.load('models/deeplog_key_model.pt', device=device)
    model.train()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Fine-tune
    print(f"\n[4/4] Fine-tuning for {epochs} epochs...")
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in tqdm(fp_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(fp_loader)
        history.append({'epoch': epoch+1, 'loss': epoch_loss})
        print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}")
    
    # Save fine-tuned model
    model.save('models/deeplog_key_model_ft.pt')
    print(f"✓ Saved fine-tuned model to models/deeplog_key_model_ft.pt")
    
    # Save update log
    log_data = {
        'false_positive_count': len(false_positive_blockids),
        'fine_tune_samples': len(fp_df),
        'fine_tune_windows': len(X_fp),
        'epochs': epochs,
        'learning_rate': lr,
        'history': history
    }
    
    with open('artifacts/online/update_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"✓ Saved update log to artifacts/online/update_log.json")
    
    return model, history


if __name__ == '__main__':
    online_finetune()
    print("\n✓ Task Done: Online fine-tuning completed")
    print("  - models/deeplog_key_model_ft.pt")
    print("  - artifacts/online/update_log.json")
