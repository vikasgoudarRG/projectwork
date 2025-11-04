# DeepLog GPU Support Guide

Generated: 2025-11-04 18:37:00

## Overview

DeepLog now supports **3 device types** for accelerated training:

1. **CUDA (NVIDIA GPUs)** - For NVIDIA graphics cards
2. **MPS (Apple Silicon)** - For Mac M1/M2/M3 chips  
3. **CPU** - Fallback for all systems

The device is automatically detected and selected based on availability.

---

## Device Detection

### Automatic Selection

The system automatically detects and uses the best available device:

```python
from src.utils.device import get_device

device = get_device(prefer_mps=True, verbose=True)
# Priority: CUDA > MPS > CPU
```

**Output example (Mac M1):**
```
✓ Using device: mps
  Apple Silicon GPU (Metal Performance Shaders)
  Note: MPS provides significant speedup over CPU on M1/M2/M3 chips
```

**Output example (NVIDIA GPU):**
```
✓ Using device: cuda
  GPU: NVIDIA GeForce RTX 3080
  Memory: 10.00 GB
```

**Output example (CPU fallback):**
```
✓ Using device: cpu
  Warning: Training on CPU will be slower. Consider using GPU if available.
```

---

## Platform-Specific Setup

### 1. Mac M1/M2/M3 (Apple Silicon)

**Requirements:**
- macOS 12.3+ (Monterey or later)
- PyTorch 1.12.0+ with MPS support

**Check MPS availability:**
```bash
python -m src.utils.device
```

**Expected speedup:**
- **Training**: 3-5x faster than CPU
- **Inference**: 2-4x faster than CPU

**Installation:**
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio
```

**Troubleshooting:**
If MPS is not detected:
```bash
# Check PyTorch version (must be 1.12+)
python -c "import torch; print(torch.__version__)"

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Update PyTorch if needed
pip install --upgrade torch
```

---

### 2. NVIDIA GPUs (CUDA)

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+ or 12.1+
- PyTorch with CUDA support

**Check CUDA availability:**
```bash
python -m src.utils.device
```

**Expected speedup:**
- **Training**: 10-20x faster than CPU
- **Inference**: 5-10x faster than CPU

**Installation:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA:**
```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Common issues:**

**Issue:** CUDA out of memory
```bash
# Solution: Reduce batch size
python -m scripts.run_training --task key --batch_size 64
```

**Issue:** CUDA version mismatch
```bash
# Solution: Reinstall PyTorch matching your CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. CPU (All Platforms)

**Default fallback** for systems without GPU support.

**Expected training time:**
- Key LSTM: ~30-60 minutes
- Value LSTM: ~30-60 minutes

**Optimization tips:**
```bash
# Reduce batch size for faster iteration (but longer overall time)
python -m scripts.run_training --task key --batch_size 64

# Reduce data size for testing
# (modify dataset_loader.py to use only a subset)
```

---

## Performance Comparison

| Device | Training Time (Key LSTM) | Speedup vs CPU |
|--------|-------------------------|----------------|
| **Mac M1 Pro** | 10-15 minutes | 3-5x |
| **Mac M2 Max** | 8-12 minutes | 5-7x |
| **NVIDIA RTX 3080** | 3-5 minutes | 10-15x |
| **NVIDIA RTX 4090** | 2-3 minutes | 15-20x |
| **CPU (8 cores)** | 30-60 minutes | 1x (baseline) |

*Actual times depend on dataset size (~575k sequences, ~30M windows)*

---

## Usage Examples

### Check Available Devices
```bash
python -m src.utils.device
```

### Training with Auto Device Selection
```bash
# Automatically uses best available device
make train-key
make train-value
```

### Force CPU (for testing)
```python
# In your script, modify:
device = get_device(prefer_mps=False, verbose=True)
# Or manually:
device = torch.device('cpu')
```

---

## MPS-Specific Notes (Mac M1/M2/M3)

### Known Limitations

1. **MPS Backend is newer** - Some operations may fall back to CPU
2. **Memory management** - Shared memory with system (not dedicated VRAM)
3. **PyTorch support** - Requires PyTorch 1.12+ (2.0+ recommended)

### Best Practices

**1. Monitor memory usage:**
```python
import torch
print(f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
```

**2. Clear cache if needed:**
```python
torch.mps.empty_cache()
```

**3. Optimize batch size:**
```bash
# Start with default (128), reduce if memory issues
python -m scripts.run_training --task key --batch_size 96
```

### Compatibility Check
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

---

## CUDA-Specific Notes (NVIDIA)

### GPU Memory Management

**Check GPU memory:**
```bash
nvidia-smi

# Or in Python:
python -c "import torch; print(torch.cuda.memory_summary())"
```

**Clear GPU cache:**
```python
torch.cuda.empty_cache()
```

**Monitor during training:**
```bash
watch -n 1 nvidia-smi
```

### Multi-GPU Support

Currently, the pipeline uses **single GPU** by default. For multi-GPU:

```python
# Modify training script to use DataParallel
model = torch.nn.DataParallel(model)
```

---

## Troubleshooting Guide

### Issue 1: Device not detected

**Symptoms:**
```
✓ Using device: cpu
  Warning: Training on CPU will be slower.
```

**Solutions:**

**For Mac M1/M2/M3:**
```bash
# Check macOS version (need 12.3+)
sw_vers

# Update PyTorch
pip install --upgrade torch

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

**For NVIDIA GPU:**
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 2: Out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
# or
RuntimeError: MPS backend out of memory
```

**Solutions:**
```bash
# 1. Reduce batch size
python -m scripts.run_training --task key --batch_size 64

# 2. Clear cache before training
python -c "import torch; torch.cuda.empty_cache()"  # CUDA
python -c "import torch; torch.mps.empty_cache()"   # MPS

# 3. Close other GPU-using applications
```

---

### Issue 3: MPS slower than expected

**Possible causes:**
1. PyTorch version too old (< 2.0)
2. Operations falling back to CPU
3. System memory pressure

**Solutions:**
```bash
# Update PyTorch to latest
pip install --upgrade torch

# Monitor system memory
Activity Monitor > Memory

# Close memory-intensive apps
```

---

## Device Utility API

### Core Functions

**Get device:**
```python
from src.utils.device import get_device

device = get_device(prefer_mps=True, verbose=True)
```

**Check device type:**
```python
from src.utils.device import is_cuda_device, is_mps_device, is_cpu_device

if is_cuda_device(device):
    print("Using NVIDIA GPU")
elif is_mps_device(device):
    print("Using Apple Silicon GPU")
else:
    print("Using CPU")
```

**Get device info:**
```python
from src.utils.device import get_device_info

info = get_device_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"MPS available: {info['mps_available']}")
```

**Move data to device:**
```python
from src.utils.device import move_to_device

X_batch = move_to_device(X_batch, device)
```

---

## Testing GPU Support

### Quick Test Script

```bash
# Test device detection
python -m src.utils.device

# Test training (first 1000 samples)
python -c "
import pandas as pd
df = pd.read_csv('Event_traces.csv', nrows=1000)
df.to_csv('/tmp/test_data.csv', index=False)
"

# Train on small dataset
python -m scripts.run_training --task key \
    --data_path /tmp/test_data.csv \
    --epochs 2 \
    --batch_size 32
```

### Expected Output

**Mac M1:**
```
✓ Using device: mps
  Apple Silicon GPU (Metal Performance Shaders)

[1/5] Loading data...
[2/5] Building dataloaders...
[3/5] Creating model...
[4/5] Training for 2 epochs...
Epoch 1/2 [Train]: 100%|████████| 32/32 [00:05<00:00,  6.2it/s]
```

**NVIDIA GPU:**
```
✓ Using device: cuda
  GPU: NVIDIA GeForce RTX 3080
  Memory: 10.00 GB

[1/5] Loading data...
[2/5] Building dataloaders...
[3/5] Creating model...
[4/5] Training for 2 epochs...
Epoch 1/2 [Train]: 100%|████████| 32/32 [00:02<00:00, 15.8it/s]
```

---

## Recommendations

### For Mac M1/M2/M3 Users

✅ **Recommended:**
- Use MPS (enabled by default)
- Batch size: 128-256
- PyTorch 2.0+

### For NVIDIA GPU Users

✅ **Recommended:**
- Use CUDA (enabled by default)
- Batch size: 256-512
- Monitor GPU memory with `nvidia-smi`

### For CPU Users

✅ **Recommended:**
- Reduce batch size to 64-128
- Use smaller dataset for testing
- Consider cloud GPU (Google Colab, AWS, etc.)

---

## Cloud GPU Options

If you don't have a local GPU:

**1. Google Colab (Free/Paid)**
- Free: T4 GPU (15 GB)
- Pro: A100 GPU (40 GB)
- Upload code and data to Colab
- Runtime → Change runtime type → GPU

**2. AWS EC2**
- g4dn instances (NVIDIA T4)
- p3 instances (NVIDIA V100)
- Pay per hour

**3. Paperspace Gradient**
- Free tier: M4000 GPU
- Pro: RTX 4000/5000

---

## Summary

✅ **Mac M1/M2/M3**: Automatic MPS support (3-5x speedup)  
✅ **NVIDIA GPU**: Automatic CUDA support (10-20x speedup)  
✅ **CPU**: Fallback for all platforms  
✅ **No code changes needed** - device auto-detected  

Run `python -m src.utils.device` to check your setup!

---

*For more information:*
- **Workflow commands**: `report/WORKFLOW_COMMANDS.md`
- **Model training**: `report/MODEL_TRAINING_README.md`
- **PyTorch MPS**: https://pytorch.org/docs/stable/notes/mps.html
- **PyTorch CUDA**: https://pytorch.org/get-started/locally/
