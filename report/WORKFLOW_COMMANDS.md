# DeepLog Workflow Commands Reference

Generated: 2025-11-04 18:32:00

## Quick Command Reference

```bash
# Training
make train-key         # Train Key LSTM only
make train-value       # Train Value LSTM only

# Detection & Evaluation (requires both models)
make detect           # Run anomaly detection
make evaluate         # Compute metrics
make visualize        # Generate workflow graph

# Online Learning
make finetune         # Fine-tune on false positives

# Pipeline
make full-pipeline    # Run complete pipeline (train → detect → eval → visualize)

# Utilities
make clean-models     # Remove all models and artifacts
```

---

## 1. Training Commands

### Train Key LSTM Only
```bash
make train-key
```

**What it does:**
- Trains the Key LSTM model for next-event prediction
- Uses window size h=10, top-g=9 recall metric
- Saves to `models/deeplog_key_model.pt`

**Output files:**
- `models/deeplog_key_model.pt` - Model checkpoint
- `artifacts/training/key_history.json` - Training history
- `artifacts/training/key_loss_curve.png` - Loss/accuracy plots

**Time:** ~30-60 minutes on CPU, ~5-10 minutes on GPU

---

### Train Value LSTM Only
```bash
make train-value
```

**What it does:**
- Trains the Value LSTM model for time-series regression
- Computes normalization stats and anomaly threshold (μ+kσ)
- Saves to `models/deeplog_value_model.pt`

**Output files:**
- `models/deeplog_value_model.pt` - Model checkpoint
- `artifacts/training/value_history.json` - Training history
- `artifacts/training/value_loss_curve.png` - MSE loss plot
- `artifacts/training/value_norm.json` - Z-score normalization stats
- `artifacts/training/value_threshold.json` - Anomaly threshold (μ+kσ)

**Time:** ~30-60 minutes on CPU, ~5-10 minutes on GPU

---

### Train Both Models
```bash
python -m scripts.run_training --task both
```

**What it does:**
- Trains Key LSTM first, then Value LSTM
- Sequential training of both models

---

## 2. Detection Commands

### Run Anomaly Detection (Requires Both Models)
```bash
make detect
```

**What it does:**
- Loads both Key LSTM and Value LSTM models
- Runs detection on test set (10% of data)
- Computes KEY anomaly (true_next ∉ top-9)
- Computes VALUE anomaly (error > μ+3σ)
- Fuses results: anomaly = KEY OR VALUE
- Aggregates per BlockId

**Output files:**
- `artifacts/detection/predictions.csv` - Per-window predictions
- `artifacts/detection/session_anomalies.csv` - Per-BlockId aggregation
- `artifacts/detection/detection_stats.json` - Detection statistics

**Prerequisites:** Both `deeplog_key_model.pt` and `deeplog_value_model.pt` must exist

**Time:** ~5-10 minutes

---

### Detection with Custom Threshold
```bash
python -m scripts.run_training --task detect --k_sigma 2.5
```

**Parameters:**
- `--k_sigma`: Threshold multiplier (default 3.0, range 2.0-3.5)
  - Lower k → more sensitive (more anomalies detected)
  - Higher k → less sensitive (fewer false positives)

**Example thresholds:**
- `k=2.0`: Very sensitive (μ + 2σ)
- `k=2.5`: Moderate
- `k=3.0`: Standard (default)
- `k=3.5`: Conservative

---

## 3. Evaluation Commands

### Evaluate Detection Results
```bash
make evaluate
```

**What it does:**
- Loads detection results
- Computes metrics: Precision, Recall, F1, Accuracy
- Generates classification report
- Creates visualization plots

**Output files:**
- `artifacts/eval/classification_report.md` - Detailed metrics
- `artifacts/eval/f1_latency_table.csv` - Performance summary
- `artifacts/eval/metrics.json` - JSON metrics
- `graphs/f1_comparison_bar.png` - F1 score visualization
- `graphs/roc_curve.png` - ROC curve

**Prerequisites:** Must run `make detect` first

**Time:** <1 minute

---

## 4. Visualization Commands

### Generate Workflow Graph
```bash
make visualize
```

**What it does:**
- Builds event transition graph from sequences
- Highlights anomalous transitions in red
- Creates NetworkX visualization

**Output files:**
- `artifacts/visual/workflow_graph.png` - Event transition graph
- `artifacts/visual/workflow_stats.json` - Graph statistics

**Prerequisites:** Must run `make detect` first

**Time:** ~2-5 minutes

---

## 5. Online Learning Commands

### Fine-tune on False Positives
```bash
make finetune
```

**What it does:**
- Identifies false positive BlockIds from detection results
- Fine-tunes Key LSTM on those samples
- Saves updated model

**Output files:**
- `models/deeplog_key_model_ft.pt` - Fine-tuned model
- `artifacts/online/update_log.json` - Fine-tuning log

**Prerequisites:** Must run `make detect` first

**Time:** ~5-10 minutes

---

## 6. Pipeline Commands

### Full Pipeline
```bash
make full-pipeline
```

**What it does:**
Runs the complete workflow in sequence:
1. Train Key LSTM
2. Train Value LSTM
3. Run detection
4. Evaluate results
5. Generate visualizations

**Time:** ~1-2 hours on CPU

---

## 7. Advanced Options

### Custom Hyperparameters

**Key LSTM training:**
```bash
python -m scripts.run_training --task key \
    --epochs 30 \
    --lr 0.0005 \
    --batch_size 256 \
    --h 15
```

**Value LSTM training:**
```bash
python -m scripts.run_training --task value \
    --epochs 25 \
    --lr 0.001 \
    --batch_size 128
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 128)
- `--h`: Window size (default: 10)
- `--data_path`: Path to Event_traces.csv (default: Event_traces.csv)

---

## 8. Utility Commands

### Clean All Artifacts
```bash
make clean-models
```

**What it does:**
- Removes all trained models
- Deletes all artifacts
- Cleans graphs directory

**Directories cleaned:**
- `models/`
- `artifacts/training/`
- `artifacts/detection/`
- `artifacts/eval/`
- `artifacts/visual/`
- `artifacts/online/`
- `graphs/`

---

## 9. File Structure After Training

```
projectwork/
├── models/
│   ├── deeplog_key_model.pt          # Key LSTM checkpoint
│   ├── deeplog_value_model.pt        # Value LSTM checkpoint
│   └── deeplog_key_model_ft.pt       # Fine-tuned model (optional)
│
├── artifacts/
│   ├── training/
│   │   ├── key_history.json          # Key training history
│   │   ├── key_loss_curve.png        # Key loss plot
│   │   ├── value_history.json        # Value training history
│   │   ├── value_loss_curve.png      # Value loss plot
│   │   ├── value_norm.json           # Normalization stats
│   │   └── value_threshold.json      # Anomaly threshold
│   │
│   ├── detection/
│   │   ├── predictions.csv           # Per-window predictions
│   │   ├── session_anomalies.csv     # Per-BlockId results
│   │   └── detection_stats.json      # Detection statistics
│   │
│   ├── eval/
│   │   ├── classification_report.md  # Detailed metrics
│   │   ├── f1_latency_table.csv      # Performance table
│   │   └── metrics.json              # JSON metrics
│   │
│   ├── visual/
│   │   ├── workflow_graph.png        # Event transition graph
│   │   └── workflow_stats.json       # Graph statistics
│   │
│   └── online/
│       └── update_log.json           # Fine-tuning log
│
└── graphs/
    ├── f1_comparison_bar.png         # F1 score visualization
    └── roc_curve.png                 # ROC curve
```

---

## 10. Common Workflows

### Scenario 1: First-time Training
```bash
# Train both models
make train-key
make train-value

# Run detection and evaluation
make detect
make evaluate
make visualize
```

---

### Scenario 2: Re-train with Different Threshold
```bash
# No need to re-train models
# Just re-run detection with new threshold
python -m scripts.run_training --task detect --k_sigma 2.5
make evaluate
```

---

### Scenario 3: Re-train from Scratch
```bash
# Clean old models
make clean-models

# Train with custom parameters
python -m scripts.run_training --task key --epochs 30 --lr 0.0005
python -m scripts.run_training --task value --epochs 25 --lr 0.001

# Run detection and evaluation
make detect
make evaluate
```

---

### Scenario 4: Quick Full Pipeline
```bash
# Run everything at once
make full-pipeline
```

---

## 11. Testing Key LSTM Only (Standalone)

**Question: Can I test Key LSTM without training Value LSTM?**

**Answer:** The standard `make detect` requires **both models** because DeepLog uses a fusion strategy (KEY OR VALUE anomaly). However, you can evaluate the Key LSTM standalone:

### Option 1: Test Key LSTM Performance (Standalone)
```bash
python -m scripts.run_training --task key
```

During training, the script already computes:
- **Top-1 Accuracy** (exact next-event prediction)
- **Top-9 Recall** (true event in top-9 predictions)

Check the output or `artifacts/training/key_history.json` for metrics.

### Option 2: Create a Key-Only Detection Script (Manual)

If you want to run Key-only detection on test data:

```python
import torch
from src.models.key_lstm import KeyLSTM, top_g_recall
from src.data.dataset_loader import load_event_traces, split_by_blockid, build_key_dataloaders

# Load data
df = load_event_traces('Event_traces.csv')
_, _, test_df = split_by_blockid(df, seed=1337)
_, _, test_loader, _ = build_key_dataloaders(
    df, df, test_df, h=10, batch_size=128
)

# Load model
device = torch.device('cpu')
model = KeyLSTM.load('models/deeplog_key_model.pt', device=device)
model.eval()

# Evaluate
total_acc = 0
total_recall = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_acc += (torch.argmax(logits, dim=1) == y_batch).float().mean().item()
        total_recall += top_g_recall(logits, y_batch, g=9)

print(f"Test Accuracy: {total_acc / len(test_loader):.4f}")
print(f"Test Top-9 Recall: {total_recall / len(test_loader):.4f}")
```

---

## 12. Troubleshooting

### Error: FileNotFoundError: models/deeplog_value_model.pt
**Solution:** Train Value LSTM first: `make train-value`

### Error: Out of Memory
**Solution:** Reduce batch size: `--batch_size 64`

### Error: Training too slow
**Solution:** Check GPU availability or reduce dataset size for testing

### Error: Poor F1 score
**Solution:** Try different threshold: `--k_sigma 2.5`

---

## Summary Table

| Command | Prerequisites | Output | Time |
|---------|--------------|--------|------|
| `make train-key` | Data files | Key LSTM model | 30-60 min |
| `make train-value` | Data files | Value LSTM model | 30-60 min |
| `make detect` | Both models | Detection results | 5-10 min |
| `make evaluate` | Detection results | Metrics & plots | <1 min |
| `make visualize` | Detection results | Workflow graph | 2-5 min |
| `make finetune` | Detection results | Fine-tuned model | 5-10 min |
| `make full-pipeline` | Data files | Everything | 1-2 hours |

---

*For more details, see:*
- **Data preprocessing**: `report/DATA_PREPROCESSING_README.md`
- **Model training**: `report/MODEL_TRAINING_README.md`
