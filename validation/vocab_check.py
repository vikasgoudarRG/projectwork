import pandas as pd
import json
import os
import numpy as np
from validation.utils import write_json, write_text, save_bar_topn


def run():
    traces_path = 'Event_traces.csv'
    templates_path = 'HDFS.log_templates.csv'
    vocab_path = 'artifacts/validation/vocab_summary.json'
    missing_ids_path = 'artifacts/validation/traces_ids_missing_in_templates.txt'
    frequencies_path = 'artifacts/validation/template_frequencies_bar.png'
    
    if not os.path.exists(traces_path):
        return {'error': 'Traces file not found'}
    
    df_traces = pd.read_csv(traces_path)
    
    sequence_col = None
    for col in df_traces.columns:
        if col.lower() in ['features', 'sequence']:
            sequence_col = col
            break
    
    if sequence_col is None:
        return {'error': 'Could not find sequence column'}
    
    all_event_ids = set()
    event_id_counts = {}
    
    for seq in df_traces[sequence_col]:
        try:
            if pd.isna(seq) or str(seq) == 'nan':
                continue
            
            seq_str = str(seq)
            if seq_str.startswith('['):
                try:
                    seq_list = json.loads(seq_str)
                except:
                    seq_list = [x.strip() for x in seq_str.strip('[]').split(',')]
            else:
                seq_list = [x.strip() for x in seq_str.split(',')]
            
            for item in seq_list:
                eid = str(item).strip().strip('"').strip("'")
                if eid:
                    all_event_ids.add(eid)
                    event_id_counts[eid] = event_id_counts.get(eid, 0) + 1
        except:
            pass
    
    event_ids_int = []
    for eid in all_event_ids:
        try:
            if eid.startswith('E'):
                eid_clean = eid[1:]
            else:
                eid_clean = eid
            event_ids_int.append(int(eid_clean))
        except:
            pass
    
    if len(event_ids_int) == 0:
        return {'error': 'No valid event IDs found'}
    
    vocab_size = len(all_event_ids)
    min_id = min(event_ids_int) if event_ids_int else 0
    max_id = max(event_ids_int) if event_ids_int else 0
    
    expected_range = max_id - min_id + 1
    sparsity = (expected_range - vocab_size) / max(expected_range, 1)
    
    dense_or_sparse = 'dense' if sparsity < 0.2 else 'sparse'
    
    problems = []
    if min_id < 0:
        problems.append('Negative IDs found')
    if sparsity > 0.5:
        problems.append(f'High sparsity: {sparsity:.2%}')
    
    template_ids = set()
    if os.path.exists(templates_path):
        df_templates = pd.read_csv(templates_path)
        id_col = None
        for col in df_templates.columns:
            if col.lower() in ['eventid', 'event_id', 'id', 'log_key']:
                id_col = col
                break
        
        if id_col:
            template_ids = set(df_templates[id_col].astype(str))
            missing_ids = []
            for eid in all_event_ids:
                if eid not in template_ids:
                    eid_clean = eid[1:] if eid.startswith('E') else eid
                    if eid_clean not in template_ids:
                        missing_ids.append(eid)
            
            write_text(missing_ids_path, '\n'.join(sorted(missing_ids)))
        else:
            write_text(missing_ids_path, '')
    else:
        write_text(missing_ids_path, '')
    
    sorted_ids = sorted(event_id_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_ids[:30]]
    values = [x[1] for x in sorted_ids[:30]]
    
    save_bar_topn(frequencies_path, labels, values, 'Top 30 Template Frequencies', top=30)
    
    summary = {
        'vocab_size': vocab_size,
        'min_id': min_id,
        'max_id': max_id,
        'dense_or_sparse': dense_or_sparse,
        'problems': problems
    }
    
    write_json(vocab_path, summary)
    
    return {
        'vocab_path': vocab_path,
        'missing_ids_path': missing_ids_path,
        'frequencies_path': frequencies_path
    }

