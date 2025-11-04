import re
import os
from validation.utils import read_text_lines, write_text, write_json, write_csv_rows


def run():
    log_path = 'HDFS.log.tmp'
    if not os.path.exists(log_path):
        log_path = 'HDFS.log'
    
    sample_path = 'artifacts/validation/raw_sample.md'
    bad_rows_path = 'artifacts/validation/bad_rows_raw.csv'
    summary_path = 'artifacts/validation/raw_blockid_summary.json'
    
    blockid_regex = re.compile(r'blk_-?\d+')
    
    total_lines = 0
    total_blockid_occurrences = 0
    unique_blockids_raw = set()
    bad_rows = []
    sample_lines = []
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                total_lines += 1
                if i < 20:
                    sample_lines.append(line.strip())
                
                matches = blockid_regex.findall(line)
                if matches:
                    total_blockid_occurrences += len(matches)
                    unique_blockids_raw.update(matches)
                else:
                    if len(bad_rows) < 100:
                        bad_rows.append({'line_number': i + 1, 'content': line.strip()[:200]})
    except Exception as e:
        pass
    
    malformed_ratio = len(bad_rows) / max(total_lines, 1)
    
    sample_content = f"""# Raw Log Sample

## First 20 Lines

```
{chr(10).join(sample_lines[:20])}
```

## Summary

- Total lines: {total_lines}
- Total BlockID occurrences: {total_blockid_occurrences}
- Unique BlockIDs: {len(unique_blockids_raw)}
- Malformed ratio: {malformed_ratio:.4f}
"""
    
    write_text(sample_path, sample_content)
    
    write_csv_rows(bad_rows_path, bad_rows, ['line_number', 'content'])
    
    summary = {
        'total_lines': total_lines,
        'total_blockid_occurrences': total_blockid_occurrences,
        'unique_blockids_raw': len(unique_blockids_raw),
        'malformed_ratio': malformed_ratio
    }
    
    write_json(summary_path, summary)
    
    with open('artifacts/validation/raw_blockids.pkl', 'wb') as f:
        import pickle
        pickle.dump(unique_blockids_raw, f)
    
    return {
        'sample_path': sample_path,
        'bad_rows_path': bad_rows_path,
        'summary_path': summary_path
    }

