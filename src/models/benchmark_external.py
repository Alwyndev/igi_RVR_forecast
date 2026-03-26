import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# ============================================================================
# ARCHITECTURE 1: Internal Model (Residual BiLSTM V1.1 / Multi)
# ============================================================================
class RVRBiLSTM_Multi(nn.Module):
    def __init__(self, input_size, hidden_size=384, num_layers=2, output_size=50, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(hidden_size)
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.layer_norm_out = nn.LayerNorm(hidden_size * 2)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.layer_norm_in(self.input_proj(x))
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        last = x2[:, -1, :] 
        last = self.layer_norm_out(last)
        return self.fc_head(last)

# ============================================================================
# ARCHITECTURE 2: External Model (Residual LSTM)
# ============================================================================
class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h_state=None):
        lstm_out, (h, c) = self.lstm(x, h_state)
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        output = lstm_out + residual
        output = self.dropout(output)
        return output, (h, c)

class MultiHorizonResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, output_size=50):
        super().__init__()
        self.lstm_blocks = nn.ModuleList()
        for i in range(num_layers):
            inp = input_size if i == 0 else hidden_size
            self.lstm_blocks.append(ResidualLSTMBlock(inp, hidden_size, dropout))
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        lstm_out = x
        for block in self.lstm_blocks:
            lstm_out, _ = block(lstm_out)
        return self.projection_head(lstm_out[:, -1, :])

# ============================================================================
# UTILITIES
# ============================================================================
def calculate_accuracy(y_true, y_pred, threshold=200):
    diff = np.abs(y_true - y_pred)
    return (diff <= threshold).mean() * 100

def run_inference(model, device, data_array, lookback=36):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(data_array) - lookback + 1):
            window = data_array[i:i+lookback]
            x_batch = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            preds.append(model(x_batch).cpu().numpy())
    return np.concatenate(preds, axis=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark starting on {device}...")
    root = r"c:\Users\alwyn\OneDrive\Desktop\IGI_Antigravity"
    data_path = os.path.join(root, "data", "processed", "igia_rvr_training_dataset_multi.parquet")
    internal_weights = os.path.join(root, "models", "best_bilstm_multi.pt")
    external_weights = os.path.join(root, "external_models", "best_model.pt")
    scaler_x_path = os.path.join(root, "data", "processed", "scalers_multi", "scaler_X.pkl")
    scaler_y_path = os.path.join(root, "data", "processed", "scalers_multi", "scaler_y.pkl")
    
    print("Loading scalers...")
    scaler_X, scaler_y = joblib.load(scaler_x_path), joblib.load(scaler_y_path)
    
    print("Loading models...")
    model_int = RVRBiLSTM_Multi(input_size=104).to(device)
    state_int = torch.load(internal_weights, map_location=device)
    model_int.load_state_dict(state_int['model_state'])
    
    model_ext = MultiHorizonResidualLSTM(input_size=104).to(device)
    
    # Robust loading for directory-based state_dict
    def load_model_weights(search_path, device):
        # Recursively look for a directory that looks like a torch save (contains data.pkl)
        for root_dir, dirs, files in os.walk(search_path):
            if "data.pkl" in files:
                print(f"Detected model weights at: {root_dir}")
                try:
                    return torch.load(root_dir, map_location=device)
                except Exception as e:
                    print(f"Failed to load from {root_dir}: {e}")
        
        # If not found in dirs, try files
        for root_dir, dirs, files in os.walk(search_path):
            for f in files:
                if f.endswith(".pt") or f.endswith(".pt.zip"):
                    try:
                        return torch.load(os.path.join(root_dir, f), map_location=device)
                    except Exception:
                        continue
        raise FileNotFoundError(f"Could not find valid model weights in {search_path}")

    state_ext = load_model_weights(os.path.join(root, "external_models"), device)
    key = 'model_state_dict' if 'model_state_dict' in state_ext else 'model_state'
    model_ext.load_state_dict(state_ext[key] if isinstance(state_ext, dict) else state_ext)

    # Load Data (Dec 2024 - Feb 2025)
    print("Loading data for standard winter window...")
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)
    
    # Window: 2024-12-01 to 2025-02-28
    # We need to include lookback window before the start date to have valid predictions for Dec 1st.
    start_eval = pd.Timestamp("2024-12-01")
    end_eval = pd.Timestamp("2025-02-28 23:50:00")
    
    # To get pred for Dec 1st 00:00, we need 35 prior steps (approx 6 hours)
    start_buffer = start_eval - pd.Timedelta(hours=6)
    
    eval_df = df[(df.index >= start_buffer) & (df.index <= end_eval)].copy()
    
    # TARGET ORDER RECONCILIATION
    # Internal model target order (Chronological per train_multi.py)
    zones = ["09_TDZ", "10_TDZ", "11_BEG", "11_TDZ", "27_TDZ", "28_TDZ", "29_BEG", "29_TDZ", "MID_2810", "MID_2911"]
    horizons_int = ["10m", "30m", "1h", "3h", "6h"]
    target_cols_int = [f"target_{z}_rvr_actual_mean_{h}" for z in zones for h in horizons_int]
    
    # External model target order (Alphabetical per notebook)
    target_cols_ext = sorted([c for c in df.columns if c.startswith("target_")])
    
    if hasattr(scaler_X, 'feature_names_in_'):
        feature_cols = list(scaler_X.feature_names_in_)
    else:
        feature_cols = [c for c in df.columns if c not in target_cols_int][:104]
    
    print(f"Using {len(feature_cols)} features as expected by scaler.")
    X_raw = eval_df[feature_cols].values
    
    # Ground truth values for each ordering
    y_raw_int = eval_df[target_cols_int].values
    y_raw_ext = eval_df[target_cols_ext].values
    
    X_scaled = scaler_X.transform(X_raw)
    
    # Run Inference
    print("Running internal model (+Chronological targets)...")
    pred_scaled_int = run_inference(model_int, device, X_scaled)
    
    print("Running external model (+Alphabetical targets)...")
    pred_scaled_ext = run_inference(model_ext, device, X_scaled)
    
    # Align targets
    y_true_int = y_raw_int[36-1:]
    y_true_ext = y_raw_ext[36-1:]
    
    # Inverse scaling for internal
    # Note: scaler_y was likely fitted on the chronological order during training
    # We need to be careful if scaler_y was fitted on a different order.
    # In train_multi.py, it was fitted on TARGET_COLS_MULTI.
    pred_int = scaler_y.inverse_transform(pred_scaled_int)
    
    # For external, we need to check how they scaled. 
    # Their notebook used joblib to load scaler_y.pkl and called transform on sorted targets.
    # So we inverse transform their preds using the same scaler.
    pred_ext = scaler_y.inverse_transform(pred_scaled_ext)
    
    timestamps = eval_df.index[36-1:]
    
    # METRICS EVALUATION
    mae_int = mean_absolute_error(y_true_int, pred_int)
    rmse_int = np.sqrt(mean_squared_error(y_true_int, pred_int))
    r2_int = r2_score(y_true_int.flatten(), pred_int.flatten())
    acc_int = calculate_accuracy(y_true_int, pred_int)
    
    mae_ext = mean_absolute_error(y_true_ext, pred_ext)
    rmse_ext = np.sqrt(mean_squared_error(y_true_ext, pred_ext))
    r2_ext = r2_score(y_true_ext.flatten(), pred_ext.flatten())
    acc_ext = calculate_accuracy(y_true_ext, pred_ext)
    
    print("\n" + "="*40)
    print(" FINAL COMPARISON TABLE (Dec 24 - Feb 25)")
    print("="*40)
    print(f"{'Metric':<10} | {'Internal':<12} | {'External':<12}")
    print("-" * 40)
    print(f"{'MAE':<10} | {mae_int:12.2f} | {mae_ext:12.2f}")
    print(f"{'RMSE':<10} | {rmse_int:12.2f} | {rmse_ext:12.2f}")
    print(f"{'R2':<10} | {r2_int:12.4f} | {r2_ext:12.4f}")
    print(f"{'Acc@200m':<10} | {acc_int:11.2f}% | {acc_ext:11.2f}%")
    print("="*40)
    
    winner = "Internal (BiLSTM)" if mae_int < mae_ext else "External (LSTM)"
    print(f"CONCLUSION: The {winner} model is better for this period.")

    # PLOT 1: Single Zone 1h Forecast Time Series
    # For internal model (Chronological: 10m, 30m, 1h, 3h, 6h)
    # Zone 1 (09_TDZ) + 1h horizon = index 2
    idx_int = 2
    
    # For external model (Alphabetical)
    # target_09_TDZ_rvr_actual_mean_1h
    idx_ext = target_cols_ext.index("target_09_TDZ_rvr_actual_mean_1h")

    plt.figure(figsize=(15, 6))
    plt.plot(timestamps, y_true_int[:, idx_int], label="Actual RVR", color='black', alpha=0.5)
    plt.plot(timestamps, pred_int[:, idx_int], label="Internal Pred (1h)", color='green')
    plt.plot(timestamps, pred_ext[:, idx_ext], label="External Pred (1h)", color='red', linestyle='--')
    plt.title("RVR Multi-Model Comparison: Zone 09_TDZ (+1h Horizon)\nWinter Period Dec 2024 - Feb 2025")
    plt.ylabel("RVR (metres)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(root, "logs", "benchmark_comparison.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    
    # PLOT 2: Error Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true_int.flatten() - pred_int.flatten(), label="Internal Error", color='green')
    sns.kdeplot(y_true_ext.flatten() - pred_ext.flatten(), label="External Error", color='red', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Error Distribution (Prediction - Actual)")
    plt.legend()
    plt.savefig(os.path.join(root, "logs", "benchmark_error_dist.png"))

if __name__ == "__main__":
    main()
