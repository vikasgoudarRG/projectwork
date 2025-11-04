"""
Dataset loader for DeepLog training.
Reads Event_traces.csv, parses JSON Features/TimeInterval,
creates sliding windows (h=10), and splits by BlockId (80/10/10).
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def load_event_traces(path='Event_traces.csv'):
    """
    Load and parse Event_traces.csv.
    
    Returns DataFrame with columns:
    - BlockId: str
    - event_ids: List[int] (parsed from Features JSON, stripped 'E')
    - time_seq: List[float] (parsed from TimeInterval JSON)
    - label_binary: int (0=Normal, 1=Anomaly)
    """
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Event_traces.csv not found at {path}")
    
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows")
    
    # Parse Features: "[E5,E22,E5,...]" → [5, 22, 5, ...]
    # Note: This is NOT valid JSON (unquoted strings), so we parse manually
    def parse_features(x):
        if pd.isna(x):
            return []
        x = str(x).strip()
        if not x or x == '' or not x.startswith('[') or not x.endswith(']'):
            return []
        try:
            # Remove brackets and split by comma
            content = x[1:-1]  # Remove [ and ]
            if not content:
                return []
            items = [item.strip() for item in content.split(',')]
            # Strip 'E' prefix: "E5" → 5
            result = []
            for eid in items:
                if eid.startswith('E'):
                    result.append(int(eid[1:]))
                elif eid.isdigit():
                    result.append(int(eid))
            return result
        except (ValueError, TypeError, AttributeError) as e:
            return []
    
    df['event_ids'] = df['Features'].apply(parse_features)
    
    # Parse TimeInterval JSON: "[0.0, 1.0, ...]" → [0.0, 1.0, ...]
    def parse_times(x):
        if pd.isna(x):
            return []
        x = str(x).strip()
        if not x or x == '' or not x.startswith('['):
            return []
        try:
            return json.loads(x)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return []
    
    df['time_seq'] = df['TimeInterval'].apply(parse_times)
    
    # Load labels from anomaly_label.csv
    label_path = path.replace('Event_traces.csv', 'anomaly_label.csv')
    label_path = label_path.replace('/tmp/test_traces.csv', 'anomaly_label.csv')  # Handle test paths
    
    if os.path.exists(label_path):
        labels_df = pd.read_csv(label_path)
        # Map: Normal=0, Anomaly=1
        labels_df['label_binary'] = labels_df['Label'].map({'Normal': 0, 'Anomaly': 1})
        # Merge labels (keep only BlockId and label_binary to avoid column conflicts)
        df = df.drop(columns=['Label'], errors='ignore')  # Drop Event_traces Label column
        df = df.merge(labels_df[['BlockId', 'label_binary']], on='BlockId', how='left')
        print(f"  Merged labels from {label_path}")
    else:
        # Fallback: use Success/Fail from Event_traces.csv Label column
        if 'Label' in df.columns:
            df['label_binary'] = df['Label'].map({'Success': 0, 'Fail': 1})
            print(f"  Using Success/Fail labels from Event_traces.csv")
        else:
            raise ValueError("No label information found")
    
    # Drop rows with missing labels
    df = df.dropna(subset=['label_binary'])
    df['label_binary'] = df['label_binary'].astype(int)
    
    # Filter out rows with empty sequences
    df['seq_len'] = df['event_ids'].apply(len)
    original_len = len(df)
    df = df[df['seq_len'] > 0].copy()
    if len(df) < original_len:
        print(f"  Filtered out {original_len - len(df)} rows with empty sequences")
    df = df.drop(columns=['seq_len'])
    
    print(f"  Parsed {len(df)} sequences")
    print(f"  Label distribution: Normal={sum(df['label_binary']==0)}, Anomaly={sum(df['label_binary']==1)}")
    
    return df[['BlockId', 'event_ids', 'time_seq', 'label_binary']]


def get_vocab_size(template_path='HDFS.log_templates.csv'):
    """Read vocab size from template file."""
    if os.path.exists(template_path):
        templates = pd.read_csv(template_path)
        vocab_size = len(templates)
        print(f"  Vocab size from templates: {vocab_size}")
        return vocab_size
    return 29  # HDFS default


def split_by_blockid(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1337):
    """
    Split by BlockId to ensure disjoint sessions in train/val/test.
    
    Returns: (train_df, val_df, test_df)
    """
    unique_blocks = df['BlockId'].unique()
    
    # Split: 80/10/10
    train_blocks, temp_blocks = train_test_split(
        unique_blocks, test_size=(val_ratio + test_ratio), random_state=seed
    )
    val_blocks, test_blocks = train_test_split(
        temp_blocks, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed
    )
    
    train_df = df[df['BlockId'].isin(train_blocks)].reset_index(drop=True)
    val_df = df[df['BlockId'].isin(val_blocks)].reset_index(drop=True)
    test_df = df[df['BlockId'].isin(test_blocks)].reset_index(drop=True)
    
    print(f"✓ Split by BlockId: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def make_key_windows(df, h=10):
    """
    Create sliding windows for Key LSTM (next-event classification).
    
    For each sequence s: for i in range(len(s)-h):
        input = s[i:i+h], label = s[i+h]
    
    Returns:
        X: List[List[int]] (input windows)
        y: List[int] (next-key labels)
        metadata: List[dict] (BlockId, window_idx for tracking)
    """
    X, y, metadata = [], [], []
    
    for idx, row in df.iterrows():
        event_ids = row['event_ids']
        block_id = row['BlockId']
        
        if len(event_ids) <= h:
            continue  # Skip short sequences
        
        for i in range(len(event_ids) - h):
            X.append(event_ids[i:i+h])
            y.append(event_ids[i+h])
            metadata.append({'BlockId': block_id, 'window_idx': i, 'label': row['label_binary']})
    
    print(f"  Created {len(X)} key windows (h={h})")
    return X, y, metadata


def make_value_windows(df, h=10, normalize=False, train_stats=None):
    """
    Create sliding windows for Value LSTM (time-series regression).
    
    Predicts next time value from previous h values.
    
    Args:
        df: DataFrame with time_seq column
        h: window size
        normalize: whether to z-score normalize
        train_stats: dict with 'mean' and 'std' from training set
    
    Returns:
        X: List[List[float]] (input windows)
        y: List[float] (next value)
        metadata: List[dict]
        stats: dict with mean/std if normalize=True
    """
    X, y, metadata = [], [], []
    
    for idx, row in df.iterrows():
        time_seq = row['time_seq']
        block_id = row['BlockId']
        
        if len(time_seq) <= h:
            continue
        
        for i in range(len(time_seq) - h):
            X.append(time_seq[i:i+h])
            y.append(time_seq[i+h])
            metadata.append({'BlockId': block_id, 'window_idx': i, 'label': row['label_binary']})
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Normalize if requested
    stats = {}
    if normalize:
        if train_stats is None:
            # Compute stats from this data (training set)
            mean = X.mean()
            std = X.std() + 1e-8
            stats = {'mean': float(mean), 'std': float(std)}
        else:
            # Use provided stats (for val/test)
            mean = train_stats['mean']
            std = train_stats['std']
            stats = train_stats
        
        X = (X - mean) / std
        y = (y - mean) / std
    
    print(f"  Created {len(X)} value windows (h={h})")
    return X.tolist(), y.tolist(), metadata, stats


class KeyDataset(Dataset):
    """PyTorch Dataset for Key LSTM."""
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ValueDataset(Dataset):
    """PyTorch Dataset for Value LSTM."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(-1)  # [B, h, 1]
        self.y = torch.FloatTensor(y).unsqueeze(-1)  # [B, 1]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_key_dataloaders(train_df, val_df, test_df, h=10, batch_size=128):
    """Build DataLoaders for Key LSTM."""
    X_train, y_train, meta_train = make_key_windows(train_df, h=h)
    X_val, y_val, meta_val = make_key_windows(val_df, h=h)
    X_test, y_test, meta_test = make_key_windows(test_df, h=h)
    
    train_loader = DataLoader(KeyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KeyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(KeyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, (meta_train, meta_val, meta_test)


def build_value_dataloaders(train_df, val_df, test_df, h=10, batch_size=128, normalize=True):
    """Build DataLoaders for Value LSTM with normalization."""
    # Train: compute stats
    X_train, y_train, meta_train, train_stats = make_value_windows(
        train_df, h=h, normalize=normalize, train_stats=None
    )
    
    # Val/Test: use train stats
    X_val, y_val, meta_val, _ = make_value_windows(
        val_df, h=h, normalize=normalize, train_stats=train_stats
    )
    X_test, y_test, meta_test, _ = make_value_windows(
        test_df, h=h, normalize=normalize, train_stats=train_stats
    )
    
    train_loader = DataLoader(ValueDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ValueDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ValueDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_stats, (meta_train, meta_val, meta_test)
