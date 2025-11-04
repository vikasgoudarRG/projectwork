# Quick Reference Guide

## Important: Validation Only

⚠️ **This pipeline validates data but does NOT transform it.**

- ✅ **Input**: CSV files in project root (`Event_traces.csv`, `anomaly_label.csv`, etc.)
- ✅ **Output**: Validation reports (PASS/FAIL, statistics)
- ❌ **NOT Output**: Transformed data, training-ready files, windows, splits

**Data transformation happens in separate training scripts** (see `report/DATA_PREPROCESSING_README.md`).

## One-Line Commands

### Run All Validations
```bash
make validate-all
```

### Run Individual Task
```bash
make validate-<task_name>
# Examples:
make validate-raw
make validate-templates
make validate-join
```

### View Final Report
```bash
cat artifacts/validation/validation_report.md
```

### Clean All Validation Artifacts
```bash
make clean-validation
```

## File Locations

| Purpose | Location |
|---------|----------|
| Main validation report | `artifacts/validation/validation_report.md` |
| Summary JSON | `artifacts/validation/summary.json` |
| Training documentation | `report/DATA_PREPROCESSING_README.md` |
| User guide | `userdocs/README.md` |

## Input Data Files (Project Root)

**Location**: All files are in the project root directory (where you run commands)

| File | Location | Purpose | Required |
|------|----------|---------|----------|
| `Event_traces.csv` | `./Event_traces.csv` | Main training data (INPUT) | ✅ Yes |
| `anomaly_label.csv` | `./anomaly_label.csv` | Ground truth labels (INPUT) | ✅ Yes |
| `HDFS.log_templates.csv` | `./HDFS.log_templates.csv` | Event template mapping (INPUT) | ✅ Yes |
| `HDFS.log` | `./HDFS.log` | Raw log file (INPUT) | ❌ Optional |
| `HDFS.log.tmp` | `./HDFS.log.tmp` | Test sample, 10k lines (INPUT) | ❌ Optional |

**Note**: These are INPUT files - the pipeline reads them but does NOT modify them.

## Task Dependency Map

```
env → raw → templates → sessions → join → labels → vocab → windows → baseline → report → docs
```

## Common Swaps

### Swap Raw Log
```bash
cp new_log.log HDFS.log
head -n 10000 HDFS.log > HDFS.log.tmp
make validate-raw
```

### Swap Traces
```bash
cp new_traces.csv Event_traces.csv
make validate-sessions validate-join validate-vocab
```

### Swap Labels
```bash
cp new_labels.csv anomaly_label.csv
make validate-labels validate-join
```

### Swap Everything
```bash
cp new/*.csv .
cp new/HDFS.log .
head -n 10000 HDFS.log > HDFS.log.tmp
make clean-validation
make validate-all
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (file not found, parsing error) |
| 2 | Hard fail (traces_count==0 or unlabeled_rate>0.05) |

## Key Files to Check

### After Running Validation

1. ✅ **First**: `artifacts/validation/validation_report.md` (PASS/FAIL status)
2. ✅ **Second**: `artifacts/validation/join_integrity.tsv` (data integrity)
3. ✅ **Third**: `artifacts/validation/summary.json` (all statistics)

### If Validation Fails

1. 🔍 Check: `artifacts/validation/bad_rows_traces.csv` (parsing errors)
2. 🔍 Check: `artifacts/validation/join_integrity.tsv` (BlockID mismatches)
3. 🔍 Check: `artifacts/validation/validation_report.md` (repair suggestions)

## Makefile Targets

| Target | Description |
|--------|-------------|
| `validate-all` | Run all validation tasks |
| `validate-env` | Bootstrap environment |
| `validate-raw` | Extract BlockIDs from raw log |
| `validate-templates` | Validate template mapping |
| `validate-sessions` | Validate event sequences |
| `validate-join` | Check BlockID integrity (critical) |
| `validate-labels` | Validate label distribution |
| `validate-vocab` | Check vocabulary |
| `validate-windows` | Validate window readiness |
| `validate-baseline` | Check occurrence matrix |
| `validate-report` | Generate final report |
| `validate-docs` | Generate documentation |
| `clean-validation` | Remove all validation artifacts |

## Common Workflows

### First-Time Setup
```bash
# 1. Install dependencies
pip install -r requirements-dev.txt

# 2. Ensure data files are in project root
ls Event_traces.csv anomaly_label.csv HDFS.log_templates.csv

# 3. Run all validations
make validate-all

# 4. Check results
cat artifacts/validation/validation_report.md
```

### Testing New Data
```bash
# 1. Backup current data
mkdir backup && cp *.csv backup/

# 2. Replace with new data
cp new_data/*.csv .

# 3. Clean old artifacts
make clean-validation

# 4. Re-run validation
make validate-all

# 5. Compare results
diff artifacts/validation/summary.json backup/summary.json
```

### Quick Check (Single Task)
```bash
# Just check templates
make validate-templates
cat artifacts/validation/template_summary.md

# Just check labels
make validate-labels
cat artifacts/validation/label_ratio.json
```

