"""
Value LSTM Model for time-series regression.
Architecture: LSTM(64) → Linear
Anomaly detection via μ + kσ threshold on reconstruction error
"""
import torch
import torch.nn as nn
import os


class ValueLSTM(nn.Module):
    """
    Value LSTM for next-value prediction (time interval regression).
    
    Args:
        input_dim: Input feature dimension (1 for scalar time)
        hidden: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(self, input_dim=1, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: predict next value
        self.fc = nn.Linear(hidden, input_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch, h, input_dim] - time sequence
        
        Returns:
            pred: [batch, input_dim] - predicted next value
        """
        # LSTM
        out, (h_n, c_n) = self.lstm(x)  # out: [batch, h, hidden]
        
        # Use last hidden state
        last_hidden = out[:, -1, :]  # [batch, hidden]
        
        # Predict next value
        pred = self.fc(last_hidden)  # [batch, input_dim]
        
        return pred
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden': self.hidden,
            'num_layers': self.num_layers
        }, path)
        print(f"✓ Saved Value LSTM to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden=checkpoint['hidden'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"✓ Loaded Value LSTM from {path}")
        return model


def compute_mse(pred, target):
    """Compute mean squared error."""
    return ((pred - target) ** 2).mean().item()
