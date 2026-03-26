"""
model_v3.py — Residual Attention LSTM (V3)
Designed to surpass the external benchmark by combining strict causal sequence modeling 
(Unidirectional LSTM) with a Temporal Attention mechanism, solving the backwards-pooling flaw of V1.1.
"""

import torch
import torch.nn as nn

class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        out, _ = self.lstm(x)
        return self.dropout(out + residual)

class TemporalAttention(nn.Module):
    """
    Computes a weighted sum of the LSTM's temporal outputs, dynamically focusing
    on critical timesteps (e.g., rapid fog onset) instead of just pooling the final state.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = self.softmax(weights)
        context = torch.sum(weights * x, dim=1)  # (batch, hidden_size)
        return context

class RVRAttentionLSTM_V3(nn.Module):
    def __init__(self, input_size, hidden_size=384, num_layers=3, output_size=50, dropout=0.3):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.blocks.append(ResidualLSTMBlock(in_sz, hidden_size, dropout))
            
        self.attention = TemporalAttention(hidden_size)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        # x is (batch, seq, hidden)
        context = self.attention(x) # (batch, hidden)
        
        return self.head(context)

if __name__ == "__main__":
    model = RVRAttentionLSTM_V3(104)
    dummy = torch.randn(32, 36, 104)
    out = model(dummy)
    print(f"Output shape (V3): {out.shape}")
    print(f"V3 Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
