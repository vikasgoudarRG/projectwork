# DeepLog Data Preprocessing Pipeline - User Guide

## Table of Contents

1. [What This Pipeline Does](#what-this-pipeline-does)
2. [Input Sources](#input-sources)
3. [Output: Validation Only](#output-validation-only)
4. [Quick Start](#quick-start)
5. [Running the Validation Pipeline](#running-the-validation-pipeline)
6. [Viewing Results](#viewing-results)
7. [Swapping Raw Data](#swapping-raw-data)
8. [Understanding Validation Reports](#understanding-validation-reports)
9. [Troubleshooting](#troubleshooting)

## What This Pipeline Does

**This pipeline is VALIDATION ONLY - it does NOT transform data.**

### Purpose
- ✅ **Validates** that your input data files are correctly formatted
- ✅ **Checks** data quality, integrity, and consistency
- ✅ **Generates** validation reports and documentation
- ❌ **Does NOT** transform data into training-ready format
- ❌ **Does NOT** create sliding windows or splits
- ❌ **Does NOT** output processed data files

### What Happens Next
After validation passes, you'll need separate **training scripts** (not included) that will:
- Read the validated `Event_traces.csv` and `anomaly_label.csv`
- Parse JSON columns and create sliding windows
- Split data into train/val/test sets
- Transform data for model training

## Input Sources

### Where Your Data Comes From

The validation pipeline reads input files from the **project root directory** (same folder where you run the commands).

```
projectwork/                    ← Project root (current directory)
├── Event_traces.csv            ← INPUT: Main training data
├── anomaly_label.csv           ← INPUT: Ground truth labels
├── HDFS.log_templates.csv      ← INPUT: Event template mapping
├── HDFS.log                    ← INPUT: Raw log (optional)
└── HDFS.log.tmp                ← INPUT: Test sample (optional)
```

### Input File Details

| File | Location | Purpose | Read By |
|------|----------|---------|---------|
| `Event_traces.csv` | `./Event_traces.csv` | Event sequences per BlockID | sessions, join, vocab, windows |
| `anomaly_label.csv` | `./anomaly_label.csv` | Labels (Normal/Anomaly) | join, labels, windows |
| `HDFS.log_templates.csv` | `./HDFS.log_templates.csv` | Event ID → template mapping | templates, vocab |
| `HDFS.log` | `./HDFS.log` | Raw log file | raw (optional) |
| `HDFS.log.tmp` | `./HDFS.log.tmp` | Test sample (10k lines) | raw (preferred if exists) |

### How to Verify Input Files

```bash
# Check if input files exist
ls -lh Event_traces.csv anomaly_label.csv HDFS.log_templates.csv

# Check file sizes (should be > 0)
wc -l Event_traces.csv anomaly_label.csv

# View first few rows to verify format
head -n 3 Event_traces.csv
head -n 3 anomaly_label.csv
```

## Output: Validation Only

### What Gets Generated

The pipeline creates **validation reports** (not transformed data):

```
artifacts/validation/           ← All validation outputs
├── validation_report.md       ← Final PASS/FAIL report
├── summary.json               ← Consolidated statistics
├── raw_blockid_summary.json   ← BlockID extraction stats
├── sequence_stats.json        ← Sequence length stats
├── label_ratio.json           ← Label distribution
├── vocab_summary.json         ← Vocabulary stats
├── join_integrity.tsv         ← BlockID integrity check
└── ... (other validation reports)

report/
└── DATA_PREPROCESSING_README.md  ← Training guide (for future scripts)
```

### What Does NOT Get Generated

❌ **No transformed data files** (no processed CSVs, no window files, no splits)
❌ **No training-ready datasets** (no train/val/test files)
❌ **No preprocessed sequences** (no parsed/cleaned data)

### Data Transformation Happens Elsewhere

The actual data transformation (parsing, windowing, splitting) will happen in **training scripts** that you'll create separately. Those scripts will:

1. **Read** the validated `Event_traces.csv` and `anomaly_label.csv`
2. **Parse** JSON columns (`Features`, `TimeInterval`)
3. **Create** sliding windows (h=10)
4. **Split** by BlockID (80/10/10)
5. **Transform** into model-ready format (arrays, tensors, etc.)

See `report/DATA_PREPROCESSING_README.md` for code examples showing how to do this transformation.

### Validation → Training Flow

```
┌─────────────────────────────────────────────────┐
│  STEP 1: VALIDATION (This Pipeline)           │
├─────────────────────────────────────────────────┤
│  Input: Event_traces.csv, anomaly_label.csv    │
│  Process: Check format, quality, integrity     │
│  Output: Validation reports (PASS/FAIL)        │
└─────────────────────────────────────────────────┘
                    ↓
              [Validation PASS?]
                    ↓ YES
┌─────────────────────────────────────────────────┐
│  STEP 2: TRAINING SCRIPTS (You create these)   │
├─────────────────────────────────────────────────┤
│  Input: Same Event_traces.csv, anomaly_label.csv│
│  Process: Parse JSON, create windows, split     │
│  Output: Training-ready data (arrays/tensors)   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 3: MODEL TRAINING                         │
├─────────────────────────────────────────────────┤
│  Input: Training-ready data from Step 2        │
│  Process: Train LSTM models                    │
│  Output: Trained models                        │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Ensure your data files are in the project root:
   - `Event_traces.csv` (required)
   - `anomaly_label.csv` (required)
   - `HDFS.log_templates.csv` (required)
   - `HDFS.log` or `HDFS.log.tmp` (optional, for raw intake validation)

### Run All Validations

```bash
# Using Makefile (recommended)
make validate-all

# Or using Python directly
python -m scripts.validate_data run --task all
```

This will run all validation tasks and generate **validation reports** (not transformed data) in `artifacts/validation/` and `report/`.

**Important**: This pipeline validates your data but does NOT transform it. The validation reports confirm your data is ready for training. Actual data transformation (parsing, windowing, splitting) happens in separate training scripts (see `report/DATA_PREPROCESSING_README.md` for examples).

## Running the Validation Pipeline

### Option 1: Using Makefile (Recommended)

The Makefile provides convenient shortcuts for all validation tasks:

```bash
# Run all validations
make validate-all

# Run individual tasks
make validate-env          # Bootstrap environment
make validate-raw         # Extract BlockIDs from raw log
make validate-templates    # Validate template mapping
make validate-sessions     # Validate event sequences
make validate-join         # Check BlockID integrity
make validate-labels       # Validate label distribution
make validate-vocab        # Check vocabulary
make validate-windows      # Validate window readiness
make validate-baseline     # Check occurrence matrix (if present)
make validate-report       # Generate final report
make validate-docs        # Generate documentation

# Clean validation artifacts
make clean-validation
```

### Option 2: Using Python CLI

```bash
# Run all tasks
python -m scripts.validate_data run --task all

# Run individual task
python -m scripts.validate_data run --task <task_name>

# Available tasks:
# - env (bootstrap)
# - raw (raw intake)
# - templates (template check)
# - sessions (session check)
# - join (join integrity)
# - labels (label check)
# - vocab (vocabulary check)
# - windows (window check)
# - baseline (baseline check)
# - report (final report)
# - docs (documentation)
```

### Task Execution Order

Tasks should be run in this order for best results:

1. **env** - Bootstrap (creates data contract)
2. **raw** - Extract BlockIDs (if using raw log)
3. **templates** - Validate templates
4. **sessions** - Validate sequences
5. **join** - Check integrity (critical)
6. **labels** - Validate labels
7. **vocab** - Check vocabulary
8. **windows** - Validate windows
9. **baseline** - Check matrix (optional)
10. **report** - Generate summary
11. **docs** - Generate documentation

**Note**: Running `--task all` executes tasks in the correct order automatically.

## Viewing Results

### Main Reports

1. **Final Validation Report** (Start here!)
   - Location: `artifacts/validation/validation_report.md`
   - Contains: PASS/FAIL status, summary statistics, repair suggestions
   - Format: Markdown (human-readable)

2. **Consolidated Summary** (JSON)
   - Location: `artifacts/validation/summary.json`
   - Contains: All validation statistics in structured format
   - Format: JSON (for programmatic access)

3. **Preprocessing Documentation**
   - Location: `report/DATA_PREPROCESSING_README.md`
   - Contains: Complete data schema, training guide, examples
   - Format: Markdown

### Individual Validation Reports

Each validation task generates specific reports:

#### Raw Intake (`validate-raw`)
- `artifacts/validation/raw_sample.md` - Sample lines from raw log
- `artifacts/validation/raw_blockid_summary.json` - BlockID extraction stats
- `artifacts/validation/bad_rows_raw.csv` - Malformed log lines

#### Template Check (`validate-templates`)
- `artifacts/validation/template_summary.md` - Template validation summary
- `artifacts/validation/unseen_templates.tsv` - Unused templates or missing IDs

#### Session Check (`validate-sessions`)
- `artifacts/validation/sequence_stats.json` - Sequence length statistics
- `artifacts/validation/short_or_long_sessions.csv` - Sessions <3 or >200 events
- `artifacts/validation/sequence_lengths_hist.png` - Histogram visualization
- `artifacts/validation/bad_rows_traces.csv` - Parsing errors

#### Join Integrity (`validate-join`) ⚠️ CRITICAL
- `artifacts/validation/join_integrity.tsv` - BlockID overlap statistics
- `artifacts/validation/orphan_blockids_in_raw.txt` - BlockIDs only in raw log
- `artifacts/validation/unlabeled_traces.txt` - Traces without labels
- `artifacts/validation/blockids_not_in_raw.txt` - Traces not in raw log

**Hard Fail Conditions**:
- If `traces_count == 0`, the pipeline will exit with code 2
- If `unlabeled_rate > 0.05`, the pipeline will exit with code 2

#### Label Check (`validate-labels`)
- `artifacts/validation/label_ratio.json` - Label distribution (Normal/Anomaly %)
- `artifacts/validation/duplicate_labels.csv` - Duplicate label entries
- `artifacts/validation/labels_without_traces.txt` - Labels without sequences

#### Vocabulary Check (`validate-vocab`)
- `artifacts/validation/vocab_summary.json` - Vocabulary size and statistics
- `artifacts/validation/traces_ids_missing_in_templates.txt` - Missing template mappings
- `artifacts/validation/template_frequencies_bar.png` - Top 30 frequencies chart

#### Window Check (`validate-windows`)
- `artifacts/validation/window_readiness.json` - Window generation readiness
- `artifacts/validation/bad_windows.csv` - Sequences too short for windows
- `artifacts/validation/split_integrity.json` - Train/val/test split integrity
- `artifacts/validation/anomalies_vs_normal_bar.png` - Label distribution chart

### Viewing Reports

#### Command Line

```bash
# View final validation report
cat artifacts/validation/validation_report.md

# View summary JSON (pretty-printed)
python -m json.tool artifacts/validation/summary.json

# View specific report
cat artifacts/validation/template_summary.md
```

#### In Browser/Editor

All `.md` files can be opened in:
- Markdown viewers (VS Code, GitHub, etc.)
- Text editors
- Web browsers with Markdown extensions

All `.png` files can be opened in:
- Image viewers
- Web browsers
- Documentation tools

## Swapping Raw Data

### Scenario 1: Replace HDFS.log with New Raw Log

If you want to validate a new raw log file:

1. **Backup current log** (optional):
```bash
cp HDFS.log HDFS.log.backup
```

2. **Replace the log file**:
```bash
# Option A: Copy your new file
cp /path/to/your/new_log.log HDFS.log

# Option B: Create a symlink
ln -sf /path/to/your/new_log.log HDFS.log
```

3. **Create test sample** (for faster testing):
```bash
# Create a 10k line sample for testing
head -n 10000 HDFS.log > HDFS.log.tmp
```

4. **Re-run validation**:
```bash
# Re-run raw intake (uses HDFS.log.tmp if present, else HDFS.log)
make validate-raw

# Re-run all validations
make validate-all
```

**Note**: The pipeline automatically uses `HDFS.log.tmp` if it exists, otherwise falls back to `HDFS.log`.

### Scenario 2: Replace Event_traces.csv

If you want to use a new processed traces file:

1. **Backup current traces**:
```bash
cp Event_traces.csv Event_traces.csv.backup
```

2. **Replace with new file**:
```bash
cp /path/to/your/new_traces.csv Event_traces.csv
```

3. **Verify format** (check first few rows):
```bash
head -n 5 Event_traces.csv
```

4. **Re-run affected validations**:
```bash
# These tasks depend on Event_traces.csv:
make validate-sessions
make validate-join
make validate-vocab
make validate-windows

# Or re-run all
make validate-all
```

**Important**: Ensure your new `Event_traces.csv` has:
- Column: `BlockId` (or `Block_id`)
- Column: `Features` (JSON string with event IDs)
- Column: `TimeInterval` (JSON string with floats)
- Column: `Label` (Success/Fail or Normal/Anomaly)

### Scenario 3: Replace anomaly_label.csv

If you want to use new labels:

1. **Backup current labels**:
```bash
cp anomaly_label.csv anomaly_label.csv.backup
```

2. **Replace with new file**:
```bash
cp /path/to/your/new_labels.csv anomaly_label.csv
```

3. **Verify format**:
```bash
head -n 5 anomaly_label.csv
```

4. **Re-run affected validations**:
```bash
# These tasks depend on anomaly_label.csv:
make validate-labels
make validate-join
make validate-windows

# Or re-run all
make validate-all
```

**Important**: Ensure your new `anomaly_label.csv` has:
- Column: `BlockId` (or `Block_id`)
- Column: `Label` (Normal/Anomaly or 0/1)

### Scenario 4: Replace HDFS.log_templates.csv

If you have new event templates:

1. **Backup current templates**:
```bash
cp HDFS.log_templates.csv HDFS.log_templates.csv.backup
```

2. **Replace with new file**:
```bash
cp /path/to/your/new_templates.csv HDFS.log_templates.csv
```

3. **Verify format**:
```bash
head -n 5 HDFS.log_templates.csv
```

4. **Re-run affected validations**:
```bash
# These tasks depend on templates:
make validate-templates
make validate-vocab

# Or re-run all
make validate-all
```

**Important**: Ensure your new `HDFS.log_templates.csv` has:
- Column: `EventId` (or `Event_id`, `log_key`, etc.)
- Column: `EventTemplate` (or `Event_template`, `template`, etc.)
- Event IDs should match format used in `Event_traces.csv` (e.g., "E1", "E2", etc.)

### Complete Data Replacement Workflow

To replace all data files at once:

```bash
# 1. Backup current data
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp *.csv backups/$(date +%Y%m%d_%H%M%S)/
cp HDFS.log* backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# 2. Replace files
cp /path/to/new/Event_traces.csv .
cp /path/to/new/anomaly_label.csv .
cp /path/to/new/HDFS.log_templates.csv .
cp /path/to/new/HDFS.log .  # optional

# 3. Create test sample (if using raw log)
head -n 10000 HDFS.log > HDFS.log.tmp

# 4. Clean old validation artifacts
make clean-validation

# 5. Re-run all validations
make validate-all
```

## Understanding Validation Reports

### Validation Report Status

The final report (`validation_report.md`) shows:

- **PASS**: Validation successful
- **FAIL**: Validation failed (check repair suggestions)
- **WARN**: Validation passed but with warnings

### Key Metrics to Check

1. **Template Count**: Should be 25-35 (HDFS ≈29)
   - Location: `template_summary.md`
   - Check: `artifacts/validation/template_summary.md`

2. **Sequence Lengths**: Check distribution
   - Location: `sequence_stats.json`
   - Visual: `sequence_lengths_hist.png`

3. **Join Integrity**: Critical for data quality
   - Location: `join_integrity.tsv`
   - Should show: Low orphan rates, high overlap

4. **Label Distribution**: Check anomaly rate
   - Location: `label_ratio.json`
   - Expected: 3-5% anomalies for HDFS

5. **Vocabulary Size**: Should match template count
   - Location: `vocab_summary.json`
   - Should be: 29 for HDFS

### Interpreting Errors

#### High Malformed Ratio
- **Symptom**: `raw_blockid_summary.json` shows `malformed_ratio > 0.1`
- **Fix**: Check BlockID regex pattern, verify log format

#### Template Count Out of Range
- **Symptom**: Template count < 25 or > 35
- **Fix**: Review template extraction logic, check for duplicates

#### Join Integrity Failures
- **Symptom**: High orphan rates in `join_integrity.tsv`
- **Fix**: Verify BlockID format consistency across files

#### Sequence Parsing Errors
- **Symptom**: Many rows in `bad_rows_traces.csv`
- **Fix**: Check JSON format in Features/TimeInterval columns

## Troubleshooting

### Common Issues

#### Issue: "No module named 'validation'"
**Solution**: Ensure you're running from project root:
```bash
cd /path/to/projectwork
python -m scripts.validate_data run --task all
```

#### Issue: "File not found" errors
**Solution**: Check file locations:
```bash
ls -la Event_traces.csv anomaly_label.csv HDFS.log_templates.csv
```

#### Issue: Validation fails with encoding errors
**Solution**: Files might have wrong encoding. Check:
```bash
file -bi Event_traces.csv
```

#### Issue: "No valid sessions found"
**Solution**: Check Features column format:
```bash
head -n 1 Event_traces.csv | cut -d',' -f4
```
Should be JSON format: `"[E5,E22,E5]"`

#### Issue: Hard fail in join_check
**Solution**: 
1. Check `artifacts/validation/join_integrity.tsv`
2. Verify BlockID format matches across all files
3. Check `unlabeled_traces.txt` for missing labels

### Getting Help

1. Check validation reports in `artifacts/validation/`
2. Review `report/DATA_PREPROCESSING_README.md` for data format details
3. Check `artifacts/validation/validation_report.md` for repair suggestions

### Debug Mode

To see more detailed output, check individual task outputs:
```bash
# Run with verbose Python output
python -m scripts.validate_data run --task raw 2>&1 | tee validation.log
```

## Next Steps

After successful validation:

1. Review `report/DATA_PREPROCESSING_README.md` for training instructions
2. Use validated data for model training
3. Check `artifacts/validation/validation_report.md` for any warnings

---

*For technical details, see `report/DATA_PREPROCESSING_README.md`*

