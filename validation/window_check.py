import pandas as pd
import json
import os
import glob
from validation.utils import write_json, write_csv_rows, save_bar_topn


def run(h=10):
    traces_path = 'Event_traces.csv'
    labels_path = 'anomaly_label.csv'
    readiness_path = 'artifacts/validation/window_readiness.json'
    bad_windows_path = 'artifacts/validation/bad_windows.csv'
    split_integrity_path = 'artifacts/validation/split_integrity.json'
    anomalies_vs_normal_path = 'artifacts/validation/anomalies_vs_normal_bar.png'
    
    if not os.path.exists(traces_path):
        return {'error': 'Traces file not found'}
    
    df_traces = pd.read_csv(traces_path)
    
    blockid_col = None
    sequence_col = None
    
    for col in df_traces.columns:
        if col.lower() in ['blockid', 'block_id']:
            blockid_col = col
        if col.lower() in ['features', 'sequence']:
            sequence_col = col
    
    if blockid_col is None or sequence_col is None:
        return {'error': 'Could not find required columns'}
    
    total_windows_expected = 0
    bad_windows = []
    
    for idx, row in df_traces.iterrows():
        try:
            seq_str = str(row[sequence_col])
            if seq_str.startswith('['):
                seq = json.loads(seq_str)
            else:
                seq = [x.strip() for x in seq_str.split(',')]
            
            seq_len = len(seq)
            windows_from_seq = max(0, seq_len - h)
            total_windows_expected += windows_from_seq
            
            if seq_len < h:
                bad_windows.append({
                    'block_id': str(row[blockid_col]),
                    'sequence_length': seq_len,
                    'issue': f'Sequence too short for window size {h}'
                })
        except Exception as e:
            bad_windows.append({
                'block_id': str(row[blockid_col])[:50] if blockid_col in row else 'N/A',
                'sequence_length': 0,
                'issue': str(e)[:100]
            })
    
    window_files = glob.glob('train_sequences.*') + glob.glob('*train*.npz') + glob.glob('*val*.npz') + glob.glob('*test*.npz')
    window_file_found = len(window_files) > 0
    
    total_windows_found = 0
    if window_file_found:
        try:
            import numpy as np
            for fpath in window_files:
                if fpath.endswith('.npz'):
                    data = np.load(fpath)
                    if 'sequences' in data:
                        total_windows_found += len(data['sequences'])
        except:
            pass
    
    tolerance_ok = abs(total_windows_found - total_windows_expected) / max(total_windows_expected, 1) < 0.1
    
    readiness = {
        'total_windows_expected': total_windows_expected,
        'window_file_found': window_file_found,
        'total_windows_found': total_windows_found,
        'tolerance_ok': tolerance_ok
    }
    
    write_json(readiness_path, readiness)
    write_csv_rows(bad_windows_path, bad_windows, ['block_id', 'sequence_length', 'issue'])
    
    train_blockids = set()
    val_blockids = set()
    test_blockids = set()
    
    train_files = glob.glob('train_sequences.*') + glob.glob('*train*.npz')
    val_files = glob.glob('val_sequences.*') + glob.glob('*val*.npz')
    test_files = glob.glob('test_sequences.*') + glob.glob('*test*.npz')
    
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        blockid_col_labels = None
        label_col = None
        
        for col in df_labels.columns:
            if col.lower() in ['blockid', 'block_id']:
                blockid_col_labels = col
            if col.lower() in ['label']:
                label_col = col
        
        if blockid_col_labels and label_col:
            anomaly_count = 0
            normal_count = 0
            
            for _, row in df_labels.iterrows():
                label_str = str(row[label_col])
                if label_str.lower() in ['anomaly', 'fail', '1']:
                    anomaly_count += 1
                elif label_str.lower() in ['normal', 'success', '0']:
                    normal_count += 1
            
            save_bar_topn(anomalies_vs_normal_path, ['Anomaly', 'Normal'], [anomaly_count, normal_count], 'Anomalies vs Normal Distribution', top=2)
    
    split_integrity = {
        'disjoint_by_blockid': len(train_blockids & val_blockids) == 0 and len(train_blockids & test_blockids) == 0 and len(val_blockids & test_blockids) == 0,
        'counts': {
            'train': len(train_blockids),
            'val': len(val_blockids),
            'test': len(test_blockids)
        }
    }
    
    write_json(split_integrity_path, split_integrity)
    
    return {
        'readiness_path': readiness_path,
        'bad_windows_path': bad_windows_path,
        'split_integrity_path': split_integrity_path,
        'anomalies_vs_normal_path': anomalies_vs_normal_path
    }

