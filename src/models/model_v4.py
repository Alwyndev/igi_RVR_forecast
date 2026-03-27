import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Learns to weight specific timesteps (e.g. fog onset window) 
    more heavily than others for the multi-horizon prediction.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        weights = torch.tanh(self.attn(lstm_output)) # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)         # (batch, seq_len, 1)
        
        # Weighted sum: (batch, hidden_size)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights

class RVRCnnLstm_V4(nn.Module):
    """
    Phase 12: Hybrid architecture using 1D-CNN as an encoder for the LSTM.
    Extracts local temporal patterns and spatial gradients before sequence processing.
    """
    def __init__(self, input_size=104, hidden_size=384, num_layers=3, output_size=50, dropout=0.3):
        super().__init__()
        
        # 1D CNN Encoder (Temporal Gradient Extractor)
        # Input shape: (batch, input_size, seq_len)
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        
        # LSTM Backbone
        # Input shape to LSTM: (batch, seq_len, 256)
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Attention Mechanism
        self.attention = TemporalAttention(hidden_size)
        
        # Deep Projection Head
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, 36, 104)
        
        # CNN Encoder requires (batch, channel, seq_len)
        x = x.transpose(1, 2) # (batch, 104, 36)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Return to (batch, seq_len, channel) for LSTM
        x = x.transpose(1, 2) # (batch, 36, 256)
        
        # LSTM forward
        out, _ = self.lstm(x) # (batch, 36, hidden_size)
        
        # Temporal Attention over the 36 CNN-encoded features
        context, _ = self.attention(out) # (batch, hidden_size)
        
        # Output project to 50 neurons
        output = self.projection(context)
        return output
