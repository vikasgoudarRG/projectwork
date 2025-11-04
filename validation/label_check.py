import pandas as pd
import os
from validation.utils import write_json, write_csv_rows, write_text


def run():
    labels_path = 'anomaly_label.csv'
    traces_path = 'Event_traces.csv'
    ratio_path = 'artifacts/validation/label_ratio.json'
    duplicates_path = 'artifacts/validation/duplicate_labels.csv'
    without_traces_path = 'artifacts/validation/labels_without_traces.txt'
    
    if not os.path.exists(labels_path):
        return {'error': 'Labels file not found'}
    
    df_labels = pd.read_csv(labels_path)
    
    blockid_col = None
    label_col = None
    
    for col in df_labels.columns:
        if col.lower() in ['blockid', 'block_id']:
            blockid_col = col
        if col.lower() in ['label']:
            label_col = col
    
    if blockid_col is None or label_col is None:
        return {'error': 'Could not find label columns'}
    
    labels_dict = {}
    duplicates = []
    
    for idx, row in df_labels.iterrows():
        bid = str(row[blockid_col])
        label_str = str(row[label_col])
        
        if label_str.lower() in ['anomaly', 'fail', '1']:
            label = 1
        elif label_str.lower() in ['normal', 'success', '0']:
            label = 0
        else:
            try:
                label = int(label_str)
            except:
                label = None
        
        if bid in labels_dict:
            duplicates.append({
                'block_id': bid,
                'label1': labels_dict[bid],
                'label2': label
            })
        else:
            labels_dict[bid] = label
    
    anomalies = sum(1 for v in labels_dict.values() if v == 1)
    normals = sum(1 for v in labels_dict.values() if v == 0)
    total = len(labels_dict)
    
    ratio = {
        'anomalies_pct': (anomalies / max(total, 1)) * 100,
        'normals_pct': (normals / max(total, 1)) * 100,
        'total_labels': total
    }
    
    write_json(ratio_path, ratio)
    write_csv_rows(duplicates_path, duplicates, ['block_id', 'label1', 'label2'])
    
    if os.path.exists(traces_path):
        df_traces = pd.read_csv(traces_path)
        blockid_col_traces = None
        for col in df_traces.columns:
            if col.lower() in ['blockid', 'block_id']:
                blockid_col_traces = col
                break
        
        if blockid_col_traces:
            traces_blockids = set(df_traces[blockid_col_traces].astype(str))
            labels_without_traces = set(labels_dict.keys()) - traces_blockids
            write_text(without_traces_path, '\n'.join(sorted(labels_without_traces)))
        else:
            write_text(without_traces_path, '')
    else:
        write_text(without_traces_path, '')
    
    return {
        'ratio_path': ratio_path,
        'duplicates_path': duplicates_path,
        'without_traces_path': without_traces_path
    }

