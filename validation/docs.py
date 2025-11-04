import os
from datetime import datetime
from validation.utils import write_text, read_text_lines


def run():
    docs_path = 'report/DATA_PREPROCESSING_README.md'
    
    sample_traces = read_text_lines('Event_traces.csv', sample_head=5, sample_rand=0)
    sample_labels = read_text_lines('anomaly_label.csv', sample_head=5, sample_rand=0)
    sample_templates = read_text_lines('HDFS.log_templates.csv', sample_head=5, sample_rand=0)
    
    content = f"""# DeepLog Data Preprocessing README

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This document describes the data preprocessing pipeline for DeepLog anomaly detection on HDFS log data. The pipeline validates raw log files, event traces, templates, and labels to ensure data quality before training.

## File Descriptions

### Required Files

- **HDFS.log**: Raw HDFS log file (1GB+). Contains log lines with BlockIDs in format `blk_-?\\d+`
- **HDFS.log_templates.csv**: Event template mapping with columns [EventId, EventTemplate]
- **Event_traces.csv**: Processed event sequences per BlockID with columns [BlockId, Label, Type, Features, TimeInterval, Latency]
- **anomaly_label.csv**: BlockID labels with columns [BlockId, Label] where Label ∈ {{Normal, Anomaly}}

### Optional Files

- **Event_occurrence_matrix.csv**: Co-occurrence matrix for event templates (optional)

## Data Flow

```
HDFS.log
    ↓ [BlockID extraction]
BlockID → Event_traces.csv
    ↓ [Sequence parsing]
Event Sequences → Windows (h=10)
    ↓ [Split 80/10/10]
Train/Val/Test → train_key.py, train_value.py
    ↓ [Inference]
detect.py → eval_all.py
```

## Schemas

### HDFS.log

**Columns**: raw_log_line (string)

**Sample rows**:
```
{sample_traces.get('head', [])[0] if sample_traces.get('head') else 'N/A'}
```

### HDFS.log_templates.csv

**Columns**: EventId (string), EventTemplate (string)

**Sample rows**:
```
{sample_templates.get('head', [])[0] if sample_templates.get('head') else 'N/A'}
```

### Event_traces.csv

**Columns**: 
- BlockId (string): Block identifier
- Label (string): Success/Fail
- Type (int): Anomaly type
- Features (string): JSON list of event IDs like ["E5","E22","E5"]
- TimeInterval (string): JSON list of time intervals
- Latency (int): Total latency

**Sample rows**:
```
{sample_traces.get('head', [])[0] if sample_traces.get('head') else 'N/A'}
```

### anomaly_label.csv

**Columns**: BlockId (string), Label (string: Normal/Anomaly)

**Sample rows**:
```
{sample_labels.get('head', [])[0] if sample_labels.get('head') else 'N/A'}
```

## Expected Bands

- **Unique templates**: 25-35 (HDFS ≈29)
- **Window size**: h=10
- **Train/Val/Test split**: 80/10/10
- **Deterministic seed**: 1337
- **BlockID regex**: `blk_-?\\d+`

## Validation Logic Summary

1. **Raw Intake**: Extract BlockIDs from HDFS.log, count occurrences, detect malformed lines
2. **Template Check**: Verify template count in range [25,35], check for placeholders (* or <*>), detect duplicates
3. **Session Check**: Validate sequence lengths, detect short (<3) or long (>200) sessions, check timestamps
4. **Join Check**: Verify BlockID consistency across raw logs, traces, and labels (HARD FAIL if traces_count==0 or unlabeled_rate>0.05)
5. **Label Check**: Verify label distribution, detect duplicates
6. **Vocab Check**: Verify vocabulary size, check for missing template mappings
7. **Window Check**: Estimate expected windows, verify split integrity (no BlockID leakage)

## Failure Modes & Repair Playbook

### BlockID Mismatch

**Symptom**: `join_integrity.tsv` shows high orphan rates

**Repair**: 
- Check BlockID regex pattern: `blk_-?\\d+`
- Verify BlockID extraction from raw log
- Ensure consistent BlockID format in traces and labels

### Template Gaps

**Symptom**: `unseen_templates.tsv` shows unused templates or missing IDs

**Repair**:
- Review template mapping completeness
- Check for event ID format mismatches (E1 vs 1)
- Verify all event IDs in traces have corresponding templates

### Window Mismatch

**Symptom**: `window_readiness.json` shows tolerance_ok=false

**Repair**:
- Verify window size h=10 matches generation code
- Check sequence length distribution
- Ensure all sequences have length >= h

### Split Leakage

**Symptom**: `split_integrity.json` shows disjoint_by_blockid=false

**Repair**:
- Ensure BlockID-based split (not random)
- Verify no BlockID appears in multiple splits
- Check split generation logic

## Notes for Future AI Agents

### File Locations

- Raw logs: `HDFS.log` (use `HDFS.log.tmp` for testing, ~10k lines)
- Processed traces: `Event_traces.csv`
- Templates: `HDFS.log_templates.csv`
- Labels: `anomaly_label.csv`
- Validation artifacts: `artifacts/validation/`
- Reports: `report/`

### Data Columns

- **Features column** in Event_traces.csv contains JSON lists of event IDs (e.g., ["E5","E22","E5"])
- **TimeInterval column** contains JSON lists of float time intervals
- **Label column** in anomaly_label.csv uses Normal/Anomaly (map to 0/1)

### Model Training

- **Key LSTM**: Feeds on event ID sequences from Features column
- **Value LSTM**: Feeds on time intervals from TimeInterval column
- **Top-g**: Uses top-9 most frequent events (g=9)
- **Anomaly threshold**: μ + kσ (mean + k*std deviation)

### Next Scripts

1. **train_key.py**: Train key LSTM on event ID sequences
2. **train_value.py**: Train value LSTM on time intervals
3. **detect.py**: Run inference on new sequences
4. **eval_all.py**: Evaluate on test set with metrics

### Assumptions

- Window size: h=10
- Top-g events: 9
- Anomaly threshold: μ + kσ (configurable k)
- Deterministic seed: 1337
- Train/Val/Test: 80/10/10 split

---

*Generated by DeepLog validation pipeline*
"""

    write_text(docs_path, content)
    
    return {
        'docs_path': docs_path
    }

