"""
benchmark_v3.py -- Side-by-side comparison of RVRAttentionLSTM_V3 and the External Model.
Evaluates on the Dec 2024 - Feb 2025 winter period.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first,
                           dropout=dropout if dropout > 0 else 0.0)
        if input_size != hidden_size:
            self.residual_proj = nn.Linear(input_size, hidden_size)
        else:
            self.residual_proj = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h_state=None):
        lstm_out, (h, c) = self.lstm(x, h_state)
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        output = self.dropout(lstm_out + residual)
        return output, (h, c)

class MultiHorizonResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 dropout=0.3, output_size=50, projection_hidden_1=512,
                 projection_hidden_2=256):
        super().__init__()
        self.lstm_blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            self.lstm_blocks.append(ResidualLSTMBlock(layer_input_size, hidden_size, dropout))
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, projection_hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_1, projection_hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_2, output_size)
        )
    
    def forward(self, x):
        h_states = [None] * len(self.lstm_blocks)
        lstm_out = x
        for layer_idx, block in enumerate(self.lstm_blocks):
            lstm_out, h_states[layer_idx] = block(lstm_out, h_states[layer_idx])
        last_output = lstm_out[:, -1, :]
        return self.projection_head(last_output)

def calculate_accuracy(y_true, y_pred, threshold=200):
    diff = np.abs(y_true - y_pred)
    acc = (diff < threshold).mean() * 100
    return acc

@torch.no_grad()
def run_inference(model, device, X_scaled, batch_size=512):
    model.eval()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    for (batch_x,) in loader:
        # We need to create rolling windows
        # But wait, run_inference should take the whole sequence if it's already windowed or handle it
        # Here we assume X_scaled is the RAW sequences and we window them
        pass

    # Correct windowing logic
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    seq_len = 36
    all_preds = []
    for i in range(0, len(X_t) - seq_len + 1, batch_size):
        batch_indices = []
        for j in range(i, min(i + batch_size, len(X_t) - seq_len + 1)):
            batch_indices.append(X_t[j : j + seq_len])
        
        if not batch_indices: break
        
        batch_x = torch.stack(batch_indices).to(device)
        out = model(batch_x)
        all_preds.append(out.cpu().numpy())
    
    return np.concatenate(all_preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark starting on {device}...")
    
    root = str(ROOT)
    v3_weights = os.path.join(root, "models", "best_lstm_v3.pt")
    external_weights_root = os.path.join(root, "external_models", "best_model.pt")
    
    scaler_x_path = os.path.join(root, "data", "processed", "scalers_v3", "scaler_X.pkl")
    scaler_y_path = os.path.join(root, "data", "processed", "scalers_v3", "scaler_y.pkl")
    
    print("Loading scalers...")
    scaler_X, scaler_y = joblib.load(scaler_x_path), joblib.load(scaler_y_path)
    
    print("Loading models...")
    model_v3 = RVRAttentionLSTM_V3(input_size=104).to(device)
    state_v3 = torch.load(v3_weights, map_location=device)
    model_v3.load_state_dict(state_v3['model_state'])
    
    model_ext = MultiHorizonResidualLSTM(input_size=104).to(device)
    
    def load_maybe_dir(path, device):
        # Prefer the original ZIP if it exists
        zip_path = path + ".zip" if not path.endswith(".zip") else path
        if os.path.exists(zip_path):
            import zipfile
            from io import BytesIO
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                # Check if it ALREADY has a subdirectory nesting
                has_sub = any('/' in name and not name.endswith('/') for name in names)
                if has_sub:
                    return torch.load(zip_path, map_location=device)
                
                # If not, add a prefix 'archive/'
                buf = BytesIO()
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as out_zf:
                    for name in names:
                        content = zf.read(name)
                        out_zf.writestr(f"archive/{name}", content)
                buf.seek(0)
                return torch.load(buf, map_location=device)
        
        if os.path.isdir(path):
            import zipfile
            from io import BytesIO
            base_dir = path
            # Find the actual dir with data.pkl
            for r, d, f in os.walk(path):
                if "data.pkl" in f:
                    base_dir = r
                    break
            
            buf = BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
                # IMPORTANT: Put everything in a subdirectory named 'archive/'
                for r, d, f in os.walk(base_dir):
                    for file in f:
                        full_f = os.path.join(r, file)
                        rel_f = os.path.relpath(full_f, base_dir)
                        # PyTorch expects archive/data.pkl, archive/version etc.
                        zf.write(full_f, f"archive/{rel_f}")
            buf.seek(0)
            return torch.load(buf, map_location=device)
        return torch.load(path, map_location=device)

    state_ext = load_maybe_dir(external_weights_root, device)
    model_ext.load_state_dict(state_ext.get('model_state_dict', state_ext))
    
    # Load Evaluation Data
    parquet_path = os.path.join(root, "data", "processed", "igia_rvr_training_dataset_multi.parquet")
    df = pd.read_parquet(parquet_path)
    # Focus on standard winter period for evaluation
    start_eval, end_eval = "2024-12-01", "2025-02-28"
    start_buffer = pd.to_datetime(start_eval) - pd.Timedelta(hours=6)
    eval_df = df[(df.index >= start_buffer) & (df.index <= end_eval)].copy()
    
    # Target columns
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = list(scaler_X.feature_names_in_)
    
    X_raw = eval_df[feature_cols].values
    y_raw = eval_df[target_cols].values
    
    X_scaled = scaler_X.transform(X_raw)
    
    # Run Inference
    print("Running V3 model...")
    pred_scaled_v3 = run_inference(model_v3, device, X_scaled)
    
    print("Running external model...")
    pred_scaled_ext = run_inference(model_ext, device, X_scaled)
    
    # Align targets
    y_true_raw = y_raw[36-1:]
    
    # Inverse scaling
    pred_v3 = scaler_y.inverse_transform(pred_scaled_v3)
    pred_ext = scaler_y.inverse_transform(pred_scaled_ext)
    pred_v3 = np.clip(pred_v3, 0, 4000)
    pred_ext = np.clip(pred_ext, 0, 4000)
    y_true = np.clip(y_true_raw, 0, 4000)
    
    timestamps = eval_df.index[36-1:]
    
    # METRICS
    mae_v3 = mean_absolute_error(y_true, pred_v3)
    rmse_v3 = np.sqrt(mean_squared_error(y_true, pred_v3))
    r2_v3 = r2_score(y_true.flatten(), pred_v3.flatten())
    acc_v3 = calculate_accuracy(y_true, pred_v3)
    
    mae_ext = mean_absolute_error(y_true, pred_ext)
    rmse_ext = np.sqrt(mean_squared_error(y_true, pred_ext))
    r2_ext = r2_score(y_true.flatten(), pred_ext.flatten())
    acc_ext = calculate_accuracy(y_true, pred_ext)
    
    print("\n" + "="*40)
    print(" FINAL COMPARISON TABLE (Dec 24 - Feb 25)")
    print("="*40)
    print(f"{'Metric':<10} | {'V3 (Attention)':<12} | {'External'}")
    print("-" * 40)
    print(f"{'MAE':<10} | {mae_v3:>12.2f} | {mae_ext:>8.2f}")
    print(f"{'RMSE':<10} | {rmse_v3:>12.2f} | {rmse_ext:>8.2f}")
    print(f"{'R2':<10} | {r2_v3:>12.4f} | {r2_ext:>8.4f}")
    print(f"{'Acc@200m':<10} | {acc_v3:>11.2f}% | {acc_ext:>7.2f}%")
    print("="*40)
    
    winner = "V3 (Attention LSTM)" if mae_v3 < mae_ext else "External (LSTM)"
    print(f"CONCLUSION: The {winner} model is better.")

    # PLOTS
    plt.figure(figsize=(15, 6))
    idx = 2 # 09_TDZ 1h
    plt.plot(timestamps, y_true[:, idx], label="Actual RVR", color='black', alpha=0.5)
    plt.plot(timestamps, pred_v3[:, idx], label="V3 Pred (1h)", color='blue')
    plt.plot(timestamps, pred_ext[:, idx], label="External Pred (1h)", color='red', linestyle='--')
    plt.title("RVR Multi-Model Comparison: Zone 09_TDZ (+1h Horizon)\nV3 vs External")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(root, "logs", "benchmark_v3_results.png"))
    print("Plot saved to logs/benchmark_v3_results.png")

if __name__ == "__main__":
    main()
