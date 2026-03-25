"""
model_v2.py -- BiLSTM + Self-Attention for IGIA RVR Prediction (V2)

Upgrades over V1:
  1. Multi-Head Self-Attention after BiLSTM (captures long-range fog correlations)
  2. Residual connections for gradient stability
  3. Larger capacity (hidden=512)
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Self-Attention."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        return self.norm(x + attn_out)     # Residual + LayerNorm


class RVRBiLSTMAttention(nn.Module):
    """
    Stacked BiLSTM + Self-Attention for multi-zone RVR forecasting.

    Architecture:
      Input -> BiLSTM (2 layers) -> Self-Attention -> Weighted Pooling -> FC -> Output

    Args:
        input_size  : Number of input features
        hidden_size : LSTM hidden units per direction (512)
        num_layers  : Number of stacked BiLSTM layers (2)
        output_size : Number of predicted runway zones (10)
        num_heads   : Attention heads (4)
        dropout     : Dropout probability (0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        output_size: int = 10,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection (stabilizes training with larger hidden)
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Stacked Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Self-Attention over the full sequence
        self.attention = SelfAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Learnable attention pooling weights
        self.pool_weight = nn.Linear(hidden_size * 2, 1)

        # Output head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, output_size) -- 10 log(RVR) predictions
        """
        # Project input
        x = self.input_proj(x)       # (batch, seq, hidden)

        # BiLSTM
        lstm_out, _ = self.bilstm(x) # (batch, seq, hidden*2)

        # Self-Attention (captures cross-timestep correlations)
        attn_out = self.attention(lstm_out)  # (batch, seq, hidden*2)

        # Attention-weighted pooling (instead of just last timestep)
        weights = torch.softmax(self.pool_weight(attn_out), dim=1)  # (batch, seq, 1)
        context = (attn_out * weights).sum(dim=1)                   # (batch, hidden*2)

        # Output projection with residual
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)   # (batch, 10)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RVRBiLSTMAttention(input_size=104, hidden_size=512, num_layers=2, output_size=10).to(device)

    dummy = torch.randn(32, 36, 104).to(device)
    out = model(dummy)
    print(f"Model output shape: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
