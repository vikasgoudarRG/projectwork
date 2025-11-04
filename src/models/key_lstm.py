"""
Key LSTM Model for next-event classification.
Architecture: Embedding → 2×LSTM(64) → Linear
Top-g recall metric (g=9)
"""
import torch
import torch.nn as nn
import os


class KeyLSTM(nn.Module):
    """
    Key LSTM for next-event prediction.
    
    Args:
        vocab_size: Number of unique event types (29 for HDFS)
        embed_dim: Embedding dimension
        hidden: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(self, vocab_size=29, embed_dim=64, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.num_layers = num_layers
        
        # Embedding: vocab+1 to handle padding (index 0)
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: predict next event (vocab+1 classes)
        self.fc = nn.Linear(hidden, vocab_size + 1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, h] - event ID sequences
        
        Returns:
            logits: [batch, vocab+1] - classification logits
        """
        # Embed
        emb = self.embedding(x)  # [batch, h, embed_dim]
        
        # LSTM
        out, (h_n, c_n) = self.lstm(emb)  # out: [batch, h, hidden]
        
        # Use last hidden state
        last_hidden = out[:, -1, :]  # [batch, hidden]
        
        # Classify
        logits = self.fc(last_hidden)  # [batch, vocab+1]
        
        return logits
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden': self.hidden,
            'num_layers': self.num_layers
        }, path)
        print(f"✓ Saved Key LSTM to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['embed_dim'],
            hidden=checkpoint['hidden'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"✓ Loaded Key LSTM from {path}")
        return model


def top_g_recall(logits, targets, g=9):
    """
    Compute top-g recall: fraction of samples where true label is in top-g predictions.
    
    Args:
        logits: [batch, vocab+1] - model predictions
        targets: [batch] - true labels
        g: number of top predictions to consider
    
    Returns:
        recall: float in [0, 1]
    """
    # Get top-g predictions
    top_g_preds = torch.topk(logits, k=g, dim=1).indices  # [batch, g]
    
    # Check if target is in top-g
    targets_expanded = targets.unsqueeze(1).expand_as(top_g_preds)  # [batch, g]
    hits = (top_g_preds == targets_expanded).any(dim=1)  # [batch]
    
    recall = hits.float().mean().item()
    return recall


def compute_accuracy(logits, targets):
    """Compute top-1 accuracy."""
    preds = torch.argmax(logits, dim=1)
    acc = (preds == targets).float().mean().item()
    return acc
