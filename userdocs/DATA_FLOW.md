# Data Flow: Input → Validation → Training

## Quick Answer to Your Questions

### 1. Where is the input source?

**Answer**: Input files are in the **project root directory** (same folder where you run `make validate-all`):

```
📁 projectwork/              ← You are here (project root)
├── 📄 Event_traces.csv      ← INPUT (main data)
├── 📄 anomaly_label.csv     ← INPUT (labels)
├── 📄 HDFS.log_templates.csv ← INPUT (templates)
└── 📄 HDFS.log              ← INPUT (optional, raw log)
```

These CSV files are your **INPUT SOURCES**. The validation pipeline reads them from this location.

### 2. Is there output for training, or just validation?

**Answer**: **Validation only** - no transformed data output.

- ✅ **Output**: Validation reports (PASS/FAIL, statistics)
- ❌ **NOT Output**: Transformed data files, training-ready datasets

**Data transformation happens in separate training scripts** (which you'll create later).

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: INPUT FILES (Project Root)                       │
├─────────────────────────────────────────────────────────────┤
│  📄 Event_traces.csv       (575k rows)                     │
│  📄 anomaly_label.csv      (575k rows)                      │
│  📄 HDFS.log_templates.csv (29 rows)                        │
│  📄 HDFS.log               (optional, 1GB+)                │
│                                                              │
│  These are your INPUT SOURCES - the pipeline reads them     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: VALIDATION PIPELINE (This Tool)                    │
├─────────────────────────────────────────────────────────────┤
│  Reads: Event_traces.csv, anomaly_label.csv, etc.          │
│  Checks: Format, quality, integrity, consistency             │
│  Outputs: Validation reports (NOT transformed data)        │
│                                                              │
│  📊 artifacts/validation/validation_report.md              │
│  📊 artifacts/validation/summary.json                       │
│  📊 artifacts/validation/sequence_stats.json                │
│  📊 ... (other validation reports)                          │
│                                                              │
│  ✅ PASS → Data is ready for training                       │
│  ❌ FAIL → Fix issues before training                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
              [Validation Status: PASS]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: DATA TRANSFORMATION (Training Scripts)            │
├─────────────────────────────────────────────────────────────┤
│  Reads: Same Event_traces.csv, anomaly_label.csv            │
│  Processes:                                                 │
│    • Parse JSON columns (Features, TimeInterval)            │
│    • Create sliding windows (h=10)                          │
│    • Split by BlockID (80/10/10)                            │
│    • Convert to arrays/tensors                              │
│  Outputs: Training-ready data (in memory or files)         │
│                                                              │
│  📝 train_key.py    (creates windows from Features)        │
│  📝 train_value.py  (creates windows from TimeInterval)     │
│                                                              │
│  This is NOT part of the validation pipeline                │
│  You create these scripts separately                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: MODEL TRAINING                                     │
├─────────────────────────────────────────────────────────────┤
│  Input: Training-ready data from Phase 3                    │
│  Process: Train LSTM models (Key LSTM, Value LSTM)        │
│  Output: Trained model files (.pth, .h5, etc.)             │
└─────────────────────────────────────────────────────────────┘
```

## What the Validation Pipeline Does

### ✅ What It Does

1. **Reads** your input CSV files from project root
2. **Validates** format, structure, and data quality
3. **Checks** data integrity (BlockID consistency, label distribution, etc.)
4. **Generates** validation reports showing PASS/FAIL status
5. **Creates** documentation for future training scripts

### ❌ What It Does NOT Do

1. **Does NOT** transform data (no parsing, windowing, splitting)
2. **Does NOT** create training-ready files
3. **Does NOT** modify your input files
4. **Does NOT** output processed data

## Where to Find Transformation Code

The **data transformation** code (parsing, windowing, splitting) is documented in:

📖 **`report/DATA_PREPROCESSING_README.md`**

This file contains:
- Complete code examples for parsing JSON columns
- How to create sliding windows
- How to split data by BlockID
- How to prepare data for training

You'll use these examples to create your training scripts (`train_key.py`, `train_value.py`, etc.).

## Summary

| Question | Answer |
|---------|--------|
| **Where is input?** | Project root: `./Event_traces.csv`, `./anomaly_label.csv`, etc. |
| **What does validation output?** | Validation reports (PASS/FAIL), statistics, documentation |
| **Does it transform data?** | ❌ No - validation only |
| **Where does transformation happen?** | In separate training scripts (you create them) |
| **Where is transformation code?** | Examples in `report/DATA_PREPROCESSING_README.md` |

---

**Next Steps**: After validation passes, create training scripts using examples from `report/DATA_PREPROCESSING_README.md`.

