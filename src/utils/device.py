"""
Device detection utility for PyTorch.
Supports CUDA (NVIDIA), MPS (Apple Silicon M1/M2/M3), and CPU.
"""
import torch


def get_device(prefer_mps=True, verbose=True):
    """
    Automatically detect and return the best available device.
    
    Priority order:
    1. CUDA (NVIDIA GPUs) - if available
    2. MPS (Apple Silicon M1/M2/M3) - if available and prefer_mps=True
    3. CPU - fallback
    
    Args:
        prefer_mps: Whether to prefer MPS over CPU on Apple Silicon (default: True)
        verbose: Print device information (default: True)
    
    Returns:
        torch.device: The selected device
    """
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print(f"✓ Using device: {device}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    
    # Check for MPS (Apple Silicon M1/M2/M3)
    if prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print(f"✓ Using device: {device}")
            print(f"  Apple Silicon GPU (Metal Performance Shaders)")
            print(f"  Note: MPS provides significant speedup over CPU on M1/M2/M3 chips")
        return device
    
    # Fallback to CPU
    device = torch.device('cpu')
    if verbose:
        print(f"✓ Using device: {device}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  Note: MPS is available but not preferred (set prefer_mps=True to use)")
        print(f"  Warning: Training on CPU will be slower. Consider using GPU if available.")
    
    return device


def move_to_device(data, device):
    """
    Move data to specified device, handling MPS-specific cases.
    
    Args:
        data: Tensor or tuple/list of tensors
        device: Target device
    
    Returns:
        Data moved to device
    """
    if isinstance(data, (tuple, list)):
        return type(data)(move_to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def is_mps_device(device):
    """Check if device is MPS (Apple Silicon)."""
    return device.type == 'mps'


def is_cuda_device(device):
    """Check if device is CUDA (NVIDIA)."""
    return device.type == 'cuda'


def is_cpu_device(device):
    """Check if device is CPU."""
    return device.type == 'cpu'


def get_device_info():
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu_available': True,
    }
    
    if info['cuda_available']:
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


if __name__ == '__main__':
    # Test device detection
    print("="*60)
    print("PyTorch Device Detection")
    print("="*60)
    
    info = get_device_info()
    print(f"\nAvailable devices:")
    print(f"  CUDA (NVIDIA):     {info['cuda_available']}")
    if info['cuda_available']:
        print(f"    → {info['cuda_device_name']} ({info['cuda_memory_gb']:.2f} GB)")
    print(f"  MPS (Apple M1/M2): {info['mps_available']}")
    print(f"  CPU:               {info['cpu_available']}")
    
    print(f"\n" + "="*60)
    print("Selected device:")
    print("="*60)
    device = get_device(prefer_mps=True, verbose=True)
    
    print(f"\n" + "="*60)
    print("PyTorch version:", torch.__version__)
    print("="*60)
