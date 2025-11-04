import pandas as pd
import os
import pickle
from validation.utils import write_csv_rows, write_text


def run():
    traces_path = 'Event_traces.csv'
    labels_path = 'anomaly_label.csv'
    raw_blockids_path = 'artifacts/validation/raw_blockids.pkl'
    integrity_path = 'artifacts/validation/join_integrity.tsv'
    orphan_raw_path = 'artifacts/validation/orphan_blockids_in_raw.txt'
    unlabeled_traces_path = 'artifacts/validation/unlabeled_traces.txt'
    blockids_not_in_raw_path = 'artifacts/validation/blockids_not_in_raw.txt'
    
    hard_fail = False
    issues = []
    
    raw_blockids = set()
    if os.path.exists(raw_blockids_path):
        with open(raw_blockids_path, 'rb') as f:
            raw_blockids = pickle.load(f)
    
    traces_blockids = set()
    if os.path.exists(traces_path):
        df_traces = pd.read_csv(traces_path)
        blockid_col = None
        for col in df_traces.columns:
            if col.lower() in ['blockid', 'block_id']:
                blockid_col = col
                break
        
        if blockid_col:
            traces_blockids = set(df_traces[blockid_col].astype(str))
    
    labels_blockids = set()
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        blockid_col = None
        for col in df_labels.columns:
            if col.lower() in ['blockid', 'block_id']:
                blockid_col = col
                break
        
        if blockid_col:
            labels_blockids = set(df_labels[blockid_col].astype(str))
    
    if len(traces_blockids) == 0:
        hard_fail = True
        issues.append('HARD FAIL: traces_count == 0')
    
    orphan_raw = raw_blockids - traces_blockids
    unlabeled_traces = traces_blockids - labels_blockids
    blockids_not_in_raw = traces_blockids - raw_blockids
    
    unlabeled_rate = len(unlabeled_traces) / max(len(traces_blockids), 1)
    if unlabeled_rate > 0.05:
        hard_fail = True
        issues.append(f'HARD FAIL: unlabeled_rate > 0.05 ({unlabeled_rate:.4f})')
    
    integrity_rows = [
        {'set': 'raw', 'count': len(raw_blockids)},
        {'set': 'traces', 'count': len(traces_blockids)},
        {'set': 'labels', 'count': len(labels_blockids)},
        {'set': 'orphan_in_raw', 'count': len(orphan_raw)},
        {'set': 'unlabeled_traces', 'count': len(unlabeled_traces)},
        {'set': 'traces_not_in_raw', 'count': len(blockids_not_in_raw)},
        {'set': 'unlabeled_rate', 'count': f'{unlabeled_rate:.4f}'}
    ]
    
    write_csv_rows(integrity_path, integrity_rows, ['set', 'count'])
    
    write_text(orphan_raw_path, '\n'.join(sorted(orphan_raw)))
    write_text(unlabeled_traces_path, '\n'.join(sorted(unlabeled_traces)))
    write_text(blockids_not_in_raw_path, '\n'.join(sorted(blockids_not_in_raw)))
    
    result = {
        'integrity_path': integrity_path,
        'orphan_raw_path': orphan_raw_path,
        'unlabeled_traces_path': unlabeled_traces_path,
        'blockids_not_in_raw_path': blockids_not_in_raw_path,
        'hard_fail': hard_fail,
        'issues': issues
    }
    
    return result

