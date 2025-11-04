import pandas as pd
import json
import os
import numpy as np
from validation.utils import write_json, write_csv_rows, save_histogram


def run():
    traces_path = 'Event_traces.csv'
    stats_path = 'artifacts/validation/sequence_stats.json'
    short_long_path = 'artifacts/validation/short_or_long_sessions.csv'
    out_of_order_path = 'artifacts/validation/out_of_order_timestamps.csv'
    histogram_path = 'artifacts/validation/sequence_lengths_hist.png'
    bad_rows_path = 'artifacts/validation/bad_rows_traces.csv'
    
    if not os.path.exists(traces_path):
        return {'error': 'Traces file not found'}
    
    df = pd.read_csv(traces_path)
    
    blockid_col = None
    sequence_col = None
    
    for col in df.columns:
        if col.lower() in ['blockid', 'block_id']:
            blockid_col = col
        if col.lower() in ['features', 'sequence']:
            sequence_col = col
    
    if blockid_col is None or sequence_col is None:
        return {'error': 'Could not find required columns'}
    
    session_lengths = []
    short_or_long = []
    bad_rows = []
    out_of_order = []
    
    for idx, row in df.iterrows():
        try:
            seq_str = str(row[sequence_col])
            if pd.isna(seq_str) or seq_str == 'nan':
                continue
            
            if seq_str.startswith('['):
                try:
                    seq = json.loads(seq_str)
                except:
                    seq = [x.strip() for x in seq_str.strip('[]').split(',')]
            else:
                seq = [x.strip() for x in seq_str.split(',')]
            
            seq_int = []
            for item in seq:
                try:
                    item_str = str(item).strip().strip('"').strip("'")
                    if item_str.startswith('E'):
                        item_str = item_str[1:]
                    if item_str:
                        seq_int.append(int(item_str))
                except:
                    pass
            
            if len(seq_int) > 0:
                length = len(seq_int)
                session_lengths.append(length)
                
                if length < 3 or length > 200:
                    short_or_long.append({
                        'block_id': str(row[blockid_col]),
                        'length': length
                    })
            
            time_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    time_col = col
                    break
            
            if time_col:
                try:
                    time_str = str(row[time_col])
                    if time_str.startswith('['):
                        times = json.loads(time_str)
                        if len(times) > 1:
                            for i in range(1, len(times)):
                                if times[i] < times[i-1]:
                                    out_of_order.append({
                                        'block_id': str(row[blockid_col]),
                                        'position': i,
                                        'prev_time': times[i-1],
                                        'curr_time': times[i]
                                    })
                                    break
                except:
                    pass
        
        except Exception as e:
            bad_rows.append({
                'row': idx + 1,
                'block_id': str(row[blockid_col])[:50] if blockid_col in row else 'N/A',
                'issue': str(e)[:100]
            })
    
    if len(session_lengths) == 0:
        return {'error': 'No valid sessions found'}
    
    lengths_array = np.array(session_lengths)
    
    stats = {
        'count_sessions': len(session_lengths),
        'len_min': int(lengths_array.min()),
        'len_max': int(lengths_array.max()),
        'len_mean': float(lengths_array.mean()),
        'len_median': float(np.median(lengths_array)),
        'pct_lt3': float(np.sum(lengths_array < 3) / len(lengths_array) * 100),
        'pct_gt200': float(np.sum(lengths_array > 200) / len(lengths_array) * 100)
    }
    
    write_json(stats_path, stats)
    write_csv_rows(short_long_path, short_or_long, ['block_id', 'length'])
    
    if len(out_of_order) > 0:
        write_csv_rows(out_of_order_path, out_of_order, ['block_id', 'position', 'prev_time', 'curr_time'])
    else:
        write_csv_rows(out_of_order_path, [], ['block_id', 'position', 'prev_time', 'curr_time'])
    
    save_histogram(histogram_path, session_lengths, 'Sequence Length Distribution', 'Sequence Length', 'Frequency')
    
    write_csv_rows(bad_rows_path, bad_rows, ['row', 'block_id', 'issue'])
    
    return {
        'stats_path': stats_path,
        'short_long_path': short_long_path,
        'out_of_order_path': out_of_order_path,
        'histogram_path': histogram_path,
        'bad_rows_path': bad_rows_path
    }

