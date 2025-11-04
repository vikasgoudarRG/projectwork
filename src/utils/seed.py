"""
Deterministic seed utilities for reproducibility.
Seed: 1337 (fixed for all experiments)
"""
import random
import numpy as np
import torch


def set_seed(seed=1337):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic mode for cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Set deterministic seed: {seed}")
