"""
model_multi.py — Multi-Horizon Residual BiLSTM for IGIA RVR

This architecture is based on the highly successful V1.1 design (Z-Score + Residuals),
but expands the output head to natively predict 50 values (10 zones x 5 time horizons).
"""

import torch
import torch.nn as nn

class RVRBiLSTM_Multi(nn.Module):
    """
    Simultaneous spatial and temporal prediction model for RVR.
    Predicts fog progression at 10m, 30m, 1h, 3h, and 6h marks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        output_size: int = 50, # 10 zones * 5 horizons
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 1. Input Projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(hidden_size)

        # 2. Sequential BiLSTM (Manual Residuals)
        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.layer_norm_out = nn.LayerNorm(hidden_size * 2)

        # 3. Deep Output Head (Wider for multi-horizon complexity)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, input_size)
        Returns: 
           (batch, 50) raw output vector mapping to (zones x horizons)
        """
        x = self.layer_norm_in(self.input_proj(x))
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        
        # Take the final temporal state
        last = x2[:, -1, :] 
        
        last = self.layer_norm_out(last)
        out = self.fc_head(last)
        
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RVRBiLSTM_Multi(input_size=104).to(device)
    dummy = torch.randn(32, 36, 104).to(device)
    out = model(dummy)
    print(f"Output shape (Multi): {out.shape} -> Expected (Batch, 50)")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
