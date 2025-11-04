import json
import os
import glob
from pathlib import Path
from validation.utils import write_json, write_text


def run():
    summary_path = 'artifacts/validation/summary.json'
    report_path = 'artifacts/validation/validation_report.md'
    
    summary = {}
    
    json_files = [
        'artifacts/validation/raw_blockid_summary.json',
        'artifacts/validation/sequence_stats.json',
        'artifacts/validation/label_ratio.json',
        'artifacts/validation/vocab_summary.json',
        'artifacts/validation/window_readiness.json',
        'artifacts/validation/split_integrity.json'
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    key = Path(json_file).stem
                    summary[key] = data
            except:
                pass
    
    write_json(summary_path, summary)
    
    pass_fail = []
    
    if 'raw_blockid_summary' in summary:
        rbs = summary['raw_blockid_summary']
        pass_fail.append(('Raw BlockID Extraction', 'PASS' if rbs.get('unique_blockids_raw', 0) > 0 else 'FAIL'))
    
    if 'sequence_stats' in summary:
        ss = summary['sequence_stats']
        pass_fail.append(('Session Sequences', 'PASS' if ss.get('count_sessions', 0) > 0 else 'FAIL'))
    
    if 'label_ratio' in summary:
        lr = summary['label_ratio']
        pass_fail.append(('Label Distribution', 'PASS' if lr.get('total_labels', 0) > 0 else 'FAIL'))
    
    if 'vocab_summary' in summary:
        vs = summary['vocab_summary']
        vocab_size = vs.get('vocab_size', 0)
        pass_fail.append(('Vocabulary Check', 'PASS' if 25 <= vocab_size <= 35 else 'FAIL'))
    
    if 'window_readiness' in summary:
        wr = summary['window_readiness']
        pass_fail.append(('Window Readiness', 'PASS' if wr.get('tolerance_ok', False) or not wr.get('window_file_found', False) else 'WARN'))
    
    if 'split_integrity' in summary:
        si = summary['split_integrity']
        pass_fail.append(('Split Integrity', 'PASS' if si.get('disjoint_by_blockid', False) else 'WARN'))
    
    repair_suggestions = []
    
    if 'raw_blockid_summary' in summary:
        rbs = summary['raw_blockid_summary']
        if rbs.get('malformed_ratio', 0) > 0.1:
            repair_suggestions.append('High malformed ratio in raw log - check BlockID regex pattern')
    
    if 'vocab_summary' in summary:
        vs = summary['vocab_summary']
        if vs.get('vocab_size', 0) < 25 or vs.get('vocab_size', 0) > 35:
            repair_suggestions.append('Vocabulary size out of expected range [25, 35]')
        if len(vs.get('problems', [])) > 0:
            repair_suggestions.append(f"Vocabulary issues: {', '.join(vs.get('problems', []))}")
    
    report_content = f"""# Validation Report

## Summary

Total validation checks: {len(pass_fail)}

## Pass/Fail Status

| Check | Status |
|-------|--------|
{chr(10).join([f"| {name} | {status} |" for name, status in pass_fail])}

## Statistics

### Raw Data
{json.dumps(summary.get('raw_blockid_summary', {}), indent=2) if 'raw_blockid_summary' in summary else 'N/A'}

### Sequences
{json.dumps(summary.get('sequence_stats', {}), indent=2) if 'sequence_stats' in summary else 'N/A'}

### Labels
{json.dumps(summary.get('label_ratio', {}), indent=2) if 'label_ratio' in summary else 'N/A'}

### Vocabulary
{json.dumps(summary.get('vocab_summary', {}), indent=2) if 'vocab_summary' in summary else 'N/A'}

### Windows
{json.dumps(summary.get('window_readiness', {}), indent=2) if 'window_readiness' in summary else 'N/A'}

## Final Verdict

{'PASS' if all('PASS' in status for _, status in pass_fail) else 'FAIL' if any('FAIL' in status for _, status in pass_fail) else 'WARN'}

## Repair Suggestions

{chr(10).join(['- ' + s for s in repair_suggestions]) if repair_suggestions else 'No issues detected.'}
"""
    
    write_text(report_path, report_content)
    
    return {
        'summary_path': summary_path,
        'report_path': report_path
    }

