import os
from pathlib import Path
from validation.utils import detect_encoding_and_newlines, read_text_lines, write_yaml, write_text


def run():
    contract_path = 'artifacts/validation/data_contract.yaml'
    readme_path = 'artifacts/validation/README.md'
    
    encodings = {}
    for fname in ['HDFS.log', 'HDFS.log_templates.csv', 'Event_traces.csv', 'anomaly_label.csv']:
        if os.path.exists(fname):
            encodings[fname] = detect_encoding_and_newlines(fname)
    
    sample_data = {}
    for fname in ['HDFS.log', 'HDFS.log_templates.csv', 'Event_traces.csv', 'anomaly_label.csv']:
        if os.path.exists(fname):
            sample_data[fname] = read_text_lines(fname, sample_head=5, sample_rand=5)
    
    contract = {
        'schemas': {
            'HDFS.log': {
                'columns': ['raw_log_line'],
                'encoding': encodings.get('HDFS.log', {}).get('encoding', 'ascii'),
                'newline': encodings.get('HDFS.log', {}).get('newline', '\n'),
                'sample_rows': sample_data.get('HDFS.log', {}).get('head', [])[:5]
            },
            'HDFS.log_templates.csv': {
                'columns': ['EventId', 'EventTemplate'],
                'encoding': encodings.get('HDFS.log_templates.csv', {}).get('encoding', 'ascii'),
                'newline': encodings.get('HDFS.log_templates.csv', {}).get('newline', '\n'),
                'sample_rows': []
            },
            'Event_traces.csv': {
                'columns': ['BlockId', 'Label', 'Type', 'Features', 'TimeInterval', 'Latency'],
                'encoding': encodings.get('Event_traces.csv', {}).get('encoding', 'ascii'),
                'newline': encodings.get('Event_traces.csv', {}).get('newline', '\n'),
                'sample_rows': []
            },
            'anomaly_label.csv': {
                'columns': ['BlockId', 'Label'],
                'encoding': encodings.get('anomaly_label.csv', {}).get('encoding', 'utf-8'),
                'sample_rows': []
            }
        },
        'invariants': {
            'blockid_regex': 'blk_-?\\d+',
            'expected_templates': {'min': 25, 'max': 35},
            'seed': 1337,
            'train_val_test_split': [80, 10, 10],
            'window_size': 10
        }
    }
    
    write_yaml(contract_path, contract)
    
    readme_content = """# Validation Artifacts

This directory contains validation reports and intermediate artifacts.

## Files

- `data_contract.yaml`: Data schemas and invariants
- `raw_sample.md`: Sample from raw log
- `raw_blockid_summary.json`: BlockID extraction summary
- `template_summary.md`: Template validation summary
- `sequence_stats.json`: Session sequence statistics
- `join_integrity.tsv`: BlockID join integrity report
- `label_ratio.json`: Label distribution
- `vocab_summary.json`: Vocabulary summary
- `window_readiness.json`: Window generation readiness
- `summary.json`: Consolidated validation summary
- `validation_report.md`: Final validation report
"""
    
    write_text(readme_path, readme_content)
    
    return {
        'contract_path': contract_path,
        'readme_path': readme_path
    }

