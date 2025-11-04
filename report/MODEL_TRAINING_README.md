# DeepLog Model Training README

Generated: 2025-11-04 11:45:00

## Overview

This document describes the DeepLog dual-model training pipeline for HDFS log anomaly detection. The implementation uses PyTorch to train two LSTM models: **Key LSTM** (next-event classification) and **Value LSTM** (time-series regression).

## Architecture

### Key LSTM Model (Next-Event Classification)

**Purpose**: Predict the next log event ID in a sequence.

**Architecture**:
```
Input: Event ID sequence [batch, h=10]
  ↓
Embedding(vocab_size+1, embed_dim=64)
  ↓
LSTM(embed_dim=64, hidden=64, num_layers=2, dropout=0.2)
  ↓
Linear(hidden=64, vocab_size+1)
  ↓
Output: Logits [batch, vocab_size+1]
```

**Training Configuration**:
- **Window size (h)**: 10
- **Embedding dimension**: 64
- **Hidden size**: 64
- **LSTM layers**: 2
- **Dropout**: 0.2
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-3)
- **Early stopping patience**: 5 epochs
- **Top-g recall**: g=9 (evaluation metric)

**Anomaly Detection**: If true next event ∉ top-g predictions → anomaly

### Value LSTM Model (Time-Series Regression)

**Purpose**: Predict the next time interval value.

**Architecture**:
```
Input: Time sequence [batch, h=10, 1]
  ↓
LSTM(input_dim=1, hidden=64, num_layers=1)
  ↓
Linear(hidden=64, output_dim=1)
  ↓
Output: Predicted next value [batch, 1]
```

**Training Configuration**:
- **Window size (h)**: 10
- **Input dimension**: 1 (scalar time)
- **Hidden size**: 64
- **LSTM layers**: 1
- **Dropout**: 0.0
- **Loss function**: MSELoss
- **Optimizer**: Adam (lr=1e-3)
- **Early stopping patience**: 5 epochs
- **Normalization**: Z-score (fit on train, apply to val/test)

**Anomaly Detection**: If reconstruction error > μ + kσ → anomaly
- **Threshold**: μ + kσ (default k=3.0, configurable k ∈ [2.0, 3.5])
- μ and σ computed on **train normal samples only**

### Fusion Strategy

**Combined Anomaly Detection**:
```
fused_anomaly = key_anomaly OR value_anomaly
```

**Session-Level Aggregation**:
- Any window in a session anomalous → entire session anomalous
- Aggregate per BlockId for final evaluation

## Data Flow

### Input Data Schema

Refer to `report/DATA_PREPROCESSING_README.md` for detailed data schemas.

**Event_traces.csv**:
- **BlockId**: Session identifier (str)
- **Features**: JSON string `"[E5,E22,E5,...]"` → parse to integer list [5, 22, 5, ...]
- **TimeInterval**: JSON string `"[0.0, 1.0, 0.0,...]"` → parse to float list
- **Label**: "Success" or "Fail" (for reference)

**anomaly_label.csv**:
- **BlockId**: Session identifier (str)
- **Label**: "Normal" (0) or "Anomaly" (1)

**HDFS.log_templates.csv**:
- **EventId**: "E1" through "E29"
- **EventTemplate**: Human-readable template
- **Vocabulary size**: 29 unique events

### Data Preprocessing Pipeline

1. **Load and Parse**:
   - Parse Features JSON → integer event IDs
   - Parse TimeInterval JSON → float time values
   - Merge labels from anomaly_label.csv

2. **Train/Val/Test Split**:
   - **Split by BlockId** (not random) to prevent leakage
   - **Ratios**: 80% train, 10% val, 10% test
   - **Seed**: 1337 (deterministic)

3. **Sliding Window Creation**:
   - **Window size**: h=10
   - For each sequence of length L:
     - Generate L-h windows
     - Input: window[i:i+h], Label: window[i+h]

4. **Normalization (Value LSTM only)**:
   - Z-score normalization: (x - μ) / σ
   - Compute μ and σ from **training set only**
   - Apply same stats to val/test sets
   - Save stats to `artifacts/training/value_norm.json`

5. **Threshold Computation (Value LSTM)**:
   - After training, compute reconstruction errors on **train normal samples**
   - Calculate μ (mean error) and σ (std error)
   - Threshold = μ + kσ (default k=3.0)
   - Save to `artifacts/training/value_threshold.json`

## File Structure

```
projectwork/
├── Event_traces.csv              # Main training data (575k rows)
├── anomaly_label.csv             # Ground truth labels
├── HDFS.log_templates.csv        # Event templates (29 events)
├── requirements-train.txt        # Python dependencies
├── Makefile                      # Training targets
├── src/
│   ├── data/
│   │   └── dataset_loader.py    # Data loading and preprocessing
│   ├── models/
│   │   ├── key_lstm.py          # Key LSTM architecture
│   │   └── value_lstm.py        # Value LSTM architecture
│   ├── training/
│   │   ├── train_key.py         # Key LSTM training script
│   │   └── train_value.py       # Value LSTM training script
│   ├── detection/
│   │   └── detect.py            # Anomaly detection
│   ├── eval/
│   │   └── evaluate.py          # Evaluation metrics
│   ├── visual/
│   │   └── workflow_visualizer.py  # Workflow graph visualization
│   ├── online/
│   │   └── update.py            # Online fine-tuning
│   └── utils/
│       └── seed.py              # Deterministic seeding
├── scripts/
│   └── run_training.py          # Main entry point
├── models/                       # Saved model checkpoints
│   ├── deeplog_key_model.pt
│   ├── deeplog_value_model.pt
│   └── deeplog_key_model_ft.pt  # Fine-tuned model
├── artifacts/
│   ├── training/                # Training artifacts
│   │   ├── key_history.json
│   │   ├── key_loss_curve.png
│   │   ├── value_history.json
│   │   ├── value_loss_curve.png
│   │   ├── value_norm.json      # Normalization stats
│   │   └── value_threshold.json # Anomaly threshold
│   ├── detection/               # Detection outputs
│   │   ├── predictions.csv      # Per-window predictions
│   │   ├── session_anomalies.csv
│   │   └── detection_stats.json
│   ├── eval/                    # Evaluation metrics
│   │   ├── classification_report.md
│   │   ├── f1_latency_table.csv
│   │   └── metrics.json
│   ├── visual/                  # Visualizations
│   │   ├── workflow_graph.png
│   │   └── workflow_stats.json
│   └── online/                  # Online learning logs
│       └── update_log.json
├── graphs/                      # Evaluation plots
│   ├── f1_comparison_bar.png
│   └── roc_curve.png
└── report/
    ├── DATA_PREPROCESSING_README.md
    └── MODEL_TRAINING_README.md  # This document
```

## Commands

### Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-train.txt
```

### Training Commands

**Train Key LSTM only**:
```bash
make train-key
# or
python -m scripts.run_training --task key
```

**Train Value LSTM only**:
```bash
make train-value
# or
python -m scripts.run_training --task value
```

**Train both models**:
```bash
python -m scripts.run_training --task both
```

**Custom hyperparameters**:
```bash
python -m scripts.run_training --task key --epochs 30 --lr 0.001 --batch_size 256
```

### Detection & Evaluation

**Run detection** (default k=3.0):
```bash
make detect
# or
python -m scripts.run_training --task detect
```

**Run detection with custom threshold**:
```bash
python -m scripts.run_training --task detect --k_sigma 2.5
```

**Evaluate results**:
```bash
make evaluate
# or
python -m scripts.run_training --task eval
```

**Generate workflow visualization**:
```bash
make visualize
# or
python -m scripts.run_training --task visual
```

### Online Learning

**Fine-tune on false positives**:
```bash
make finetune
# or
python -m scripts.run_training --task online
```

### Full Pipeline

**Run complete pipeline** (train → detect → evaluate → visualize):
```bash
make full-pipeline
```

### Cleanup

**Remove all model artifacts**:
```bash
make clean-models
```

## Hyperparameters Summary

| Parameter | Key LSTM | Value LSTM |
|-----------|----------|------------|
| Window size (h) | 10 | 10 |
| Embedding dim | 64 | - |
| Hidden size | 64 | 64 |
| LSTM layers | 2 | 1 |
| Dropout | 0.2 | 0.0 |
| Batch size | 128 | 128 |
| Learning rate | 1e-3 | 1e-3 |
| Optimizer | Adam | Adam |
| Early stopping | 5 epochs | 5 epochs |
| Max epochs | 20 | 20 |
| Top-g | 9 | - |
| Threshold (k) | - | 3.0 |

## Output Artifacts

### Training Artifacts

**Key LSTM**:
- `models/deeplog_key_model.pt`: Model checkpoint
- `artifacts/training/key_history.json`: Training history (loss, accuracy, top-g recall)
- `artifacts/training/key_loss_curve.png`: Loss and accuracy curves

**Value LSTM**:
- `models/deeplog_value_model.pt`: Model checkpoint
- `artifacts/training/value_history.json`: Training history (MSE)
- `artifacts/training/value_loss_curve.png`: MSE curve
- `artifacts/training/value_norm.json`: Normalization statistics (μ, σ)
- `artifacts/training/value_threshold.json`: Anomaly threshold (μ+kσ)

### Detection Artifacts

- `artifacts/detection/predictions.csv`: Per-window predictions with columns:
  - window_idx, block_id, true_label, y_true, y_pred_topg
  - key_anomaly, value_error, value_anomaly, fused_anomaly
- `artifacts/detection/session_anomalies.csv`: Session-level aggregation
- `artifacts/detection/detection_stats.json`: Detection statistics

### Evaluation Artifacts

- `artifacts/eval/classification_report.md`: Detailed classification report
- `artifacts/eval/f1_latency_table.csv`: Performance summary table
- `artifacts/eval/metrics.json`: JSON metrics (accuracy, precision, recall, F1)
- `graphs/f1_comparison_bar.png`: F1 score bar chart
- `graphs/roc_curve.png`: ROC curve

### Visualization Artifacts

- `artifacts/visual/workflow_graph.png`: Event transition graph (red = anomalous transitions)
- `artifacts/visual/workflow_stats.json`: Graph statistics

### Online Learning Artifacts

- `models/deeplog_key_model_ft.pt`: Fine-tuned model
- `artifacts/online/update_log.json`: Fine-tuning log

## How to Re-train and Re-evaluate

### Scenario 1: Re-train with Different Hyperparameters

```bash
# Clean previous models
make clean-models

# Train with custom hyperparameters
python -m scripts.run_training --task key --epochs 30 --lr 0.0005 --batch_size 256
python -m scripts.run_training --task value --epochs 25 --lr 0.001

# Re-run detection and evaluation
make detect
make evaluate
```

### Scenario 2: Re-train with Different Threshold (k)

```bash
# No need to re-train models
# Just re-run detection with different k
python -m scripts.run_training --task detect --k_sigma 2.0
make evaluate
```

### Scenario 3: Fine-tune on False Positives

```bash
# After initial training and detection
python -m scripts.run_training --task online

# Re-run detection with fine-tuned model
# (requires code modification to load deeplog_key_model_ft.pt instead)
```

## Performance Metrics

Expected metrics on HDFS dataset (based on DeepLog paper):

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 0.95 - 0.99 |
| Precision | 0.90 - 0.98 |
| Recall | 0.85 - 0.95 |
| F1 Score | 0.90 - 0.96 |
| Latency | < 1 ms/log |

Actual metrics will be saved to `artifacts/eval/metrics.json` after running evaluation.

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
python -m scripts.run_training --task key --batch_size 64
```

### Issue: Training too slow

**Solution**: Check GPU availability
```python
import torch
print(torch.cuda.is_available())  # Should be True for GPU training
```

### Issue: Poor performance (low F1)

**Possible causes**:
1. Insufficient training epochs → increase `--epochs`
2. Wrong threshold k → try different `--k_sigma` values (2.0 to 3.5)
3. Data imbalance → check label distribution in data loader logs

### Issue: Missing data files

**Solution**: Ensure data files are in project root:
- `Event_traces.csv`
- `anomaly_label.csv`
- `HDFS.log_templates.csv`

Refer to `report/DATA_PREPROCESSING_README.md` for data format details.

## Notes for Future AI Agents

### Critical Data Assumptions

1. **Event_traces.csv format**:
   - **Features column**: JSON string `"[E5,E22,E5,...]"` → must parse and strip 'E' prefix
   - **TimeInterval column**: JSON string `"[0.0, 1.0, ...]"` → parse to float list
   - **BlockId**: Used for disjoint train/val/test split (NO LEAKAGE)

2. **Split strategy**:
   - **MUST split by BlockId**, not by row index
   - Ratios: 80% train, 10% val, 10% test
   - Seed: 1337 (deterministic)

3. **Window creation**:
   - Sliding window size h=10
   - For sequence [1,2,3,4,5,6,7,8,9,10,11,12]:
     - Window 0: [1..10] → label 11
     - Window 1: [2..11] → label 12
     - Window 2: [3..12] → (no label, skip)

4. **Normalization (Value LSTM)**:
   - **MUST** use train statistics for val/test normalization
   - Save stats to `artifacts/training/value_norm.json`
   - Apply same normalization during detection

5. **Threshold computation (Value LSTM)**:
   - Compute on **train NORMAL samples only** (label=0)
   - Threshold = mean_error + k * std_error
   - Default k=3.0, configurable

6. **Top-g recall (Key LSTM)**:
   - g=9 most likely predictions
   - Anomaly if true label NOT in top-9

7. **Fusion rule**:
   - Window-level: key_anomaly OR value_anomaly
   - Session-level: ANY window anomalous → session anomalous

### File Locations (Absolute Paths)

All paths relative to project root: `/home/vikasgoudar/Documents/serene-meadows/windsurf/projectwork/`

- **Training data**: `./Event_traces.csv`
- **Labels**: `./anomaly_label.csv`
- **Templates**: `./HDFS.log_templates.csv`
- **Models**: `./models/`
- **Artifacts**: `./artifacts/`
- **Graphs**: `./graphs/`

### Expected Statistics

After preprocessing (from validation reports):
- **Total sequences**: ~575,000
- **Vocabulary size**: 29 events (E1-E29)
- **Anomaly rate**: ~3-5%
- **Sequence lengths**: Min ~20, Max ~200, Mean ~60
- **Total windows**: ~30M+

### Model Loading Pattern

```python
# Key LSTM
from src.models.key_lstm import KeyLSTM
model = KeyLSTM.load('models/deeplog_key_model.pt', device='cuda')

# Value LSTM
from src.models.value_lstm import ValueLSTM
model = ValueLSTM.load('models/deeplog_value_model.pt', device='cuda')
```

### Detection Pattern

```python
from src.detection.detect import detect_anomalies

# Run detection with custom threshold
predictions_df, session_df = detect_anomalies(
    data_path='Event_traces.csv',
    h=10,
    batch_size=128,
    g=9,
    k_sigma=3.0
)
```

### Evaluation Pattern

```python
from src.eval.evaluate import evaluate_detection

metrics = evaluate_detection()
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

## References

- **DeepLog Paper**: "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning" (2017)
- **Dataset**: HDFS log dataset from Loghub
- **Data Preprocessing**: See `report/DATA_PREPROCESSING_README.md`

---

*Generated by DeepLog training pipeline*
*Timestamp: 2025-11-04 11:45:00*
