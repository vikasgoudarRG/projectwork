"""
DeepLog anomaly detection.
- KEY anomaly: true_next ∉ top-g predictions
- VALUE anomaly: error > μ + kσ
- Fused: KEY OR VALUE anomaly
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.dataset_loader import (
    load_event_traces, split_by_blockid, make_key_windows, make_value_windows,
    KeyDataset, ValueDataset
)
from src.models.key_lstm import KeyLSTM
from src.models.value_lstm import ValueLSTM
from torch.utils.data import DataLoader


def detect_anomalies(data_path='Event_traces.csv', h=10, batch_size=128, g=9, k_sigma=3.0):
    """
    Run anomaly detection on test set.
    
    Args:
        data_path: Path to Event_traces.csv
        h: Window size
        batch_size: Batch size for inference
        g: Top-g for key model
        k_sigma: Threshold multiplier (μ + k*σ)
    
    Returns:
        predictions_df: Per-window predictions
        session_df: Per-BlockId aggregated anomalies
    """
    # Setup
    set_seed(1337)
    device = get_device(prefer_mps=True, verbose=True)
    
    # Create output directories
    os.makedirs('artifacts/detection', exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading test data...")
    df = load_event_traces(data_path)
    _, _, test_df = split_by_blockid(df, seed=1337)
    print(f"  Test set: {len(test_df)} sessions")
    
    # Load models
    print("\n[2/5] Loading trained models...")
    key_model = KeyLSTM.load('models/deeplog_key_model.pt', device=device)
    value_model = ValueLSTM.load('models/deeplog_value_model.pt', device=device)
    key_model.eval()
    value_model.eval()
    
    # Load value normalization and threshold
    with open('artifacts/training/value_norm.json', 'r') as f:
        value_stats = json.load(f)
    
    with open('artifacts/training/value_threshold.json', 'r') as f:
        threshold_data = json.load(f)
    
    threshold = threshold_data['mean'] + k_sigma * threshold_data['std']
    print(f"  Value anomaly threshold (μ+{k_sigma}σ): {threshold:.6f}")
    
    # Build windows
    print("\n[3/5] Creating test windows...")
    X_key, y_key, meta_key = make_key_windows(test_df, h=h)
    X_value, y_value, meta_value, _ = make_value_windows(
        test_df, h=h, normalize=True, train_stats=value_stats
    )
    
    # Create dataloaders
    key_loader = DataLoader(KeyDataset(X_key, y_key), batch_size=batch_size, shuffle=False)
    value_loader = DataLoader(ValueDataset(X_value, y_value), batch_size=batch_size, shuffle=False)
    
    # Run detection
    print(f"\n[4/5] Running detection (top-g={g}, k={k_sigma})...")
    
    key_results = []
    value_results = []
    
    start_time = time.time()
    
    # Key model detection
    with torch.no_grad():
        for X_batch, y_batch in tqdm(key_loader, desc="Key detection"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = key_model(X_batch)
            
            # Get top-g predictions
            top_g_preds = torch.topk(logits, k=g, dim=1).indices  # [batch, g]
            
            # Check if true label is in top-g
            for i in range(len(y_batch)):
                true_label = y_batch[i].item()
                pred_set = top_g_preds[i].cpu().tolist()
                is_anomaly = true_label not in pred_set
                key_results.append({
                    'true': true_label,
                    'top_g': pred_set,
                    'key_anomaly': int(is_anomaly)
                })
    
    # Value model detection
    with torch.no_grad():
        for X_batch, y_batch in tqdm(value_loader, desc="Value detection"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = value_model(X_batch)
            
            # Compute per-sample MSE
            error = ((pred - y_batch) ** 2).mean(dim=1).cpu().numpy()  # [batch]
            
            for i in range(len(error)):
                is_anomaly = error[i] > threshold
                value_results.append({
                    'error': float(error[i]),
                    'value_anomaly': int(is_anomaly)
                })
    
    detection_time = time.time() - start_time
    latency_per_log = (detection_time / len(key_results)) * 1000  # ms per log
    
    print(f"  Detection completed in {detection_time:.2f}s")
    print(f"  Latency: {latency_per_log:.4f} ms/log")
    
    # Combine results
    print("\n[5/5] Aggregating results...")
    
    # Per-window predictions
    predictions = []
    for i in range(len(key_results)):
        pred = {
            'window_idx': i,
            'block_id': meta_key[i]['BlockId'],
            'true_label': meta_key[i]['label'],
            'y_true': key_results[i]['true'],
            'y_pred_topg': str(key_results[i]['top_g']),
            'key_anomaly': key_results[i]['key_anomaly'],
            'value_error': value_results[i]['error'],
            'value_anomaly': value_results[i]['value_anomaly'],
            'fused_anomaly': max(key_results[i]['key_anomaly'], value_results[i]['value_anomaly'])
        }
        predictions.append(pred)
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv('artifacts/detection/predictions.csv', index=False)
    print(f"✓ Saved per-window predictions to artifacts/detection/predictions.csv")
    
    # Aggregate by BlockId (any window anomalous → block anomalous)
    session_agg = predictions_df.groupby('block_id').agg({
        'true_label': 'first',
        'key_anomaly': 'max',
        'value_anomaly': 'max',
        'fused_anomaly': 'max'
    }).reset_index()
    
    session_agg.to_csv('artifacts/detection/session_anomalies.csv', index=False)
    print(f"✓ Saved session-level predictions to artifacts/detection/session_anomalies.csv")
    
    # Print examples
    anomalous = session_agg[session_agg['fused_anomaly'] == 1]
    print(f"\n  Detected {len(anomalous)} anomalous sessions (out of {len(session_agg)})")
    print(f"\n  Example anomalous BlockIds:")
    for bid in anomalous['block_id'].head(5).values:
        print(f"    - {bid}")
    
    # Save detection stats
    stats = {
        'total_windows': len(predictions_df),
        'total_sessions': len(session_agg),
        'key_anomalies': int(predictions_df['key_anomaly'].sum()),
        'value_anomalies': int(predictions_df['value_anomaly'].sum()),
        'fused_anomalies': int(predictions_df['fused_anomaly'].sum()),
        'anomalous_sessions': int(anomalous['fused_anomaly'].sum()),
        'detection_time_sec': detection_time,
        'latency_ms_per_log': latency_per_log,
        'threshold_k': k_sigma,
        'top_g': g
    }
    
    with open('artifacts/detection/detection_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return predictions_df, session_agg


if __name__ == '__main__':
    detect_anomalies()
    print("\n✓ Task Done: Detection completed")
    print("  - artifacts/detection/predictions.csv")
    print("  - artifacts/detection/session_anomalies.csv")
    print("  - artifacts/detection/detection_stats.json")
