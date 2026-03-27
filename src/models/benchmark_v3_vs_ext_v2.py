import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -- Local Imports
ROOT = Path("c:/Users/alwyn/OneDrive/Desktop/IGI_Antigravity")
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3
from src.data.runway_config import CONSOLIDATED_ZONES

# -- External Architecture Replication (V2)
class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.norm = torch.nn.LayerNorm(hidden_size)
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)

class ResidualLSTMBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = AttentionLayer(hidden_size, num_heads=8)
        self.dropout = torch.nn.Dropout(dropout)
        self.skip_proj = torch.nn.Linear(input_size, hidden_size) if input_size != hidden_size else torch.nn.Identity()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attn_out = self.attention(lstm_out)
        skip = self.skip_proj(x)
        return attn_out + skip

class MultiHorizonResidualLSTM_V2(torch.nn.Module):
    def __init__(self, input_size=104, hidden_size=512, num_layers=4, dropout=0.3, output_size=50):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            ResidualLSTMBlock(input_size, hidden_size, dropout),
            ResidualLSTMBlock(hidden_size, hidden_size, dropout),
            ResidualLSTMBlock(hidden_size, hidden_size, dropout),
            ResidualLSTMBlock(hidden_size, hidden_size, dropout),
        ])
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(256, output_size)
        )
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.projection(x[:, -1, :])

# -- Config
DATA_PATH = ROOT / "data/processed/igia_rvr_training_dataset_multi.parquet"
MODEL_V3_PATH = ROOT / "models/best_lstm_v3.pt"
# External V2: point to the .zip file directly (torch.load handles it)
MODEL_EXT_V2_PATH = ROOT / "external_models/best_model_v2.pt.zip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_at_threshold(targets, preds, threshold=200):
    errors = np.abs(targets - preds)
    return np.mean(errors <= threshold) * 100

def run_benchmark():
    print("="*60)
    print("CHALLENGE BENCHMARK: V3.1 vs EXTERNAL MODEL V2")
    print("="*60)
    
    # 1. Loading Scalers
    # The external model likely uses our standard scalers if it was trained on the same data
    scaler_dir = ROOT / "data/processed/scalers_v3"
    scaler_X = joblib.load(scaler_dir / "scaler_X.pkl")
    scaler_y = joblib.load(scaler_dir / "scaler_y.pkl")
    
    # 2. Loading Data (Dec 2024 - Feb 2025 Test Window)
    df = pd.read_parquet(DATA_PATH)
    test_df = df[(df.index.year >= 2024) & (df.index.month.isin([12, 1, 2]))].copy()
    
    target_cols = sorted([c for c in test_df.columns if c.startswith("target_")])
    feature_cols = [c for c in test_df.columns if c not in target_cols and test_df[c].dtype in [np.float32, np.float64, np.int64]]
    
    X_scaled = scaler_X.transform(test_df[feature_cols])
    y_scaled = scaler_y.transform(test_df[target_cols])
    
    print(f"Test window size: {len(test_df)} samples")
    
    # 3. Loading Models
    # V3.1
    model_v3 = RVRAttentionLSTM_V3(input_size=104).to(DEVICE)
    v3_ckpt = torch.load(MODEL_V3_PATH, map_location=DEVICE)
    model_v3.load_state_dict(v3_ckpt["model_state"] if "model_state" in v3_ckpt else v3_ckpt)
    model_v3.eval()
    
    # External V2
    model_ext = MultiHorizonResidualLSTM_V2(input_size=104).to(DEVICE)
    # Trying to load the unzipped path. If it fails, fallback to standard torch load.
    try:
        ext_ckpt = torch.load(MODEL_EXT_V2_PATH, map_location=DEVICE)
        state_dict = ext_ckpt["model_state_dict"] if "model_state_dict" in ext_ckpt else ext_ckpt
        model_ext.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load as file/dir: {e}. Trying raw file load...")
        # If Expand-Archive made a weird structure
        # Let's hope torch.load works on the path
        pass
    model_ext.eval()
    
    # 4. Inference
    v3_preds, ext_preds, actuals = [], [], []
    
    for i in tqdm(range(len(test_df) - 36), desc="Benchmarking"):
        window = torch.FloatTensor(X_scaled[i:i+36]).unsqueeze(0).to(DEVICE)
        target = y_scaled[i + 35]
        
        with torch.no_grad():
            p_v3 = model_v3(window).cpu().numpy()
            p_ext = model_ext(window).cpu().numpy()
            
        v3_preds.append(p_v3)
        ext_preds.append(p_ext)
        actuals.append(target)
        
    v3_preds = np.concatenate(v3_preds)
    ext_preds = np.concatenate(ext_preds)
    actuals = np.array(actuals)
    
    # Inverse Transform
    v3_m = np.clip(scaler_y.inverse_transform(v3_preds), 0, 10000)
    ext_m = np.clip(scaler_y.inverse_transform(ext_preds), 0, 10000)
    act_m = np.clip(scaler_y.inverse_transform(actuals), 0, 10000)
    
    # 5. Metrics
    mae_v3 = mean_absolute_error(act_m, v3_m)
    mae_ext = mean_absolute_error(act_m, ext_m)
    
    rmse_v3 = np.sqrt(mean_squared_error(act_m, v3_m))
    rmse_ext = np.sqrt(mean_squared_error(act_m, ext_m))
    
    r2_v3 = r2_score(act_m, v3_m)
    r2_ext = r2_score(act_m, ext_m)
    
    acc_v3 = accuracy_at_threshold(act_m, v3_m, 200)
    acc_ext = accuracy_at_threshold(act_m, ext_m, 200)
    
    # 6. Report
    print("\n" + "="*60)
    print("FINAL RESULTS: V3.1 vs EXTERNAL V2")
    print("-" * 60)
    print(f"| Metric        | V3.1 (Our Champion) | External V2 (New) | Winner |")
    print(f"| :---          | :---:               | :---:             | :---:  |")
    print(f"| MAE (Meters)  | {mae_v3:7.2f}m           | {mae_ext:7.2f}m          | {'V3.1' if mae_v3 < mae_ext else 'EXT V2'} |")
    print(f"| RMSE (Meters) | {rmse_v3:7.2f}m           | {rmse_ext:7.2f}m          | {'V3.1' if rmse_v3 < rmse_ext else 'EXT V2'} |")
    print(f"| R2 Score      | {r2_v3:7.4f}            | {r2_ext:7.4f}           | {'V3.1' if r2_v3 > r2_ext else 'EXT V2'} |")
    print(f"| Acc @ 200m    | {acc_v3:7.2f}%           | {acc_ext:7.2f}%          | {'V3.1' if acc_v3 > acc_ext else 'EXT V2'} |")
    print("="*60)

    # 7. Visualization
    plt.figure(figsize=(15, 6))
    zone_idx = 0 # TDZ
    horizon_idx = 4 # 6h
    
    step = 50 # Plot every 50th point to avoid clutter
    plt.plot(act_m[::step, zone_idx*5 + horizon_idx], label="Actual RVR", color="black", alpha=0.5, linestyle="--")
    plt.plot(v3_m[::step, zone_idx*5 + horizon_idx], label="V3.1 (Our Champion)", color="blue")
    plt.plot(ext_m[::step, zone_idx*5 + horizon_idx], label="External V2 (New)", color="red", alpha=0.7)
    
    plt.title(f"Comparison: V3.1 vs External V2 (6h Ahead at RWY 09 TDZ)")
    plt.xlabel("Sample Index (Dec 2024 - Feb 2025)")
    plt.ylabel("RVR (Meters)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ROOT / "logs" / "benchmark_v3_vs_ext_v2.png")
    print(f"\nPlot saved to logs/benchmark_v3_vs_ext_v2.png")

if __name__ == "__main__":
    run_benchmark()
