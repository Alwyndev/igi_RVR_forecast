"""
model.py — Stacked BiLSTM Architecture for IGIA RVR Prediction

Architecture:
  Input → 2× Bidirectional LSTM (hidden=256, dropout=0.2) → FC Layer → 10 RVR outputs

The model predicts RVR 6 hours ahead for all 10 consolidated runway zones simultaneously.
"""

import torch
import torch.nn as nn


class RVRBiLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM for multi-zone RVR forecasting.

    Args:
        input_size   : Number of input features (126 after targets removed)
        hidden_size  : LSTM hidden units per direction (256)
        num_layers   : Number of stacked BiLSTM layers (2)
        output_size  : Number of predicted runway zones (10)
        dropout      : Dropout between layers (0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_size: int = 10,
        dropout: float = 0.2,
    ):
        super(RVRBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stacked Bidirectional LSTM
        # dropout is applied between layers (not after the last layer)
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer Normalisation on the LSTM output for training stability
        # BiLSTM outputs hidden_size * 2 because of forward + backward
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Projection head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) — padded sequence of features

        Returns:
            out: (batch, output_size) — 10 RVR predictions (6h ahead)
        """
        # BiLSTM pass
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden*2)

        # Take only the last time-step's output (many-to-one)
        last = lstm_out[:, -1, :]  # (batch, hidden*2)

        # Normalise + regularise
        last = self.layer_norm(last)
        last = self.dropout(last)

        # Project to 10 RVR zones
        out = self.fc(last)  # (batch, 10)
        return out


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RVRBiLSTM(input_size=126, hidden_size=256, num_layers=2, output_size=10).to(device)

    dummy = torch.randn(32, 36, 126).to(device)  # batch=32, seq=36 steps (6h), feat=126
    out = model(dummy)
    print(f"Model output shape: {out.shape}")  # Expected: torch.Size([32, 10])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
