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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3
from src.models.model_v4 import RVRCnnLstm_V4

# -- Config
DATA_PATH = ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"
MODEL_V3_PATH = ROOT / "models" / "best_lstm_v3.pt"
MODEL_V4_PATH = ROOT / "models" / "best_lstm_v4.pt"
SCALER_V3_DIR = ROOT / "data" / "processed" / "scalers_v3"
SCALER_V4_DIR = ROOT / "data" / "processed" / "scalers_v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accuracy(targets, preds, threshold=100):
    errors = np.abs(targets - preds)
    return np.mean(errors <= threshold) * 100

def run_comparison():
    print("="*60)
    print("SIDE-BY-SIDE COMPARISON: V3.1 (LSTM) vs V4 (CNN-LSTM)")
    print("="*60)
    
    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols and df[c].dtype in [np.float32, np.float64, np.int64]]
    
    # Split: 2025 Test Set (Unseen)
    test_df = df[df.index.year == 2025].copy()
    
    # 2. Load Models & Scalers
    # V3.1
    scaler_X_v3 = joblib.load(SCALER_V3_DIR / "scaler_X.pkl")
    scaler_y_v3 = joblib.load(SCALER_V3_DIR / "scaler_y.pkl")
    model_v3 = RVRAttentionLSTM_V3(input_size=104).to(DEVICE)
    model_v3.load_state_dict(torch.load(MODEL_V3_PATH, map_location=DEVICE)["model_state"])
    model_v3.eval()
    
    # V4
    scaler_X_v4 = joblib.load(SCALER_V4_DIR / "scaler_X.pkl")
    scaler_y_v4 = joblib.load(SCALER_V4_DIR / "scaler_y.pkl")
    model_v4 = RVRCnnLstm_V4(input_size=104).to(DEVICE)
    model_v4.load_state_dict(torch.load(MODEL_V4_PATH, map_location=DEVICE)["model_state"])
    model_v4.eval()
    
    # Pre-scale features
    X_v3 = scaler_X_v3.transform(test_df[feature_cols])
    y_v3 = scaler_y_v3.transform(test_df[target_cols])
    X_v4 = scaler_X_v4.transform(test_df[feature_cols])
    y_v4 = scaler_y_v4.transform(test_df[target_cols])
    
    # 3. Inference
    v3_preds, v4_preds, actuals = [], [], []
    
    # Note: Using y_v3 as base actuals (since we clip 0-10000, they are the same in meters anyway)
    # We will use scaler_y_v3 for inverse transform of actuals for consistency
    
    for i in tqdm(range(len(test_df) - 36), desc="Comparing Models"):
        win_v3 = torch.FloatTensor(X_v3[i : i+36]).unsqueeze(0).to(DEVICE)
        win_v4 = torch.FloatTensor(X_v4[i : i+36]).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            p3 = model_v3(win_v3).cpu().numpy()
            p4 = model_v4(win_v4).cpu().numpy()
            
        v3_preds.append(p3)
        v4_preds.append(p4)
        actuals.append(y_v3[i + 35])
        
    v3_preds = np.concatenate(v3_preds)
    v4_preds = np.concatenate(v4_preds)
    actuals = np.array(actuals)
    
    # Inverse Transform to Meters
    v3_m = np.clip(scaler_y_v3.inverse_transform(v3_preds), 0, 10000)
    v4_m = np.clip(scaler_y_v4.inverse_transform(v4_preds), 0, 10000)
    act_m = np.clip(scaler_y_v3.inverse_transform(actuals), 0, 10000)
    
    # 4. Metrics
    metrics = {}
    for name, p_m in [("V3.1 (LSTM)", v3_m), ("V4 (Hybrid)", v4_m)]:
        mae = mean_absolute_error(act_m, p_m)
        rmse = np.sqrt(mean_squared_error(act_m, p_m))
        r2 = r2_score(act_m, p_m)
        acc100 = get_accuracy(act_m, p_m, 100)
        acc200 = get_accuracy(act_m, p_m, 200)
        metrics[name] = [mae, rmse, r2, acc100, acc200]
        
    # 5. Output Table
    print("\n" + "="*80)
    print(f"| Metric          | V3.1 (Attention LSTM) | V4 (CNN-LSTM Hybrid) | Winner |")
    print(f"| :---            | :---:                | :---:                | :---:  |")
    
    m_keys = ["MAE (Meters)", "RMSE (Meters)", "R2 Score", "Acc @ 100m", "Acc @ 200m"]
    for i, key in enumerate(m_keys):
        v3_val = metrics["V3.1 (LSTM)"][i]
        v4_val = metrics["V4 (Hybrid)"][i]
        
        # Decide winner
        if key in ["R2 Score", "Acc @ 100m", "Acc @ 200m"]:
            winner = "V3.1" if v3_val > v4_val else "V4"
        else:
            winner = "V3.1" if v3_val < v4_val else "V4"
            
        print(f"| {key:15} | {v3_val:18.2f} | {v4_val:18.2f} | {winner:6} |")
        
    print("="*80)
    
    # 6. Model Complexity
    p3 = sum(p.numel() for p in model_v3.parameters())
    p4 = sum(p.numel() for p in model_v4.parameters())
    print(f"Parameters: V3.1 = {p3:,} | V4 = {p4:,} (+{(p4-p3)/p3*100:.1f}%)")
    
    # 7. Charting
    plt.figure(figsize=(15, 6))
    zone_idx = 0 # 09 TDZ
    horizon_idx = 4 # 6h ahead
    
    target_idx = zone_idx * 5 + horizon_idx
    
    # Plot last 1000 samples for visibility
    plot_samples = 1000
    plt.plot(act_m[-plot_samples:, target_idx], label="Actual RVR", color="black", alpha=0.3, linestyle="--")
    plt.plot(v3_m[-plot_samples:, target_idx], label="V3.1 Prediction", color="blue", alpha=0.8)
    plt.plot(v4_m[-plot_samples:, target_idx], label="V4 Prediction", color="red", alpha=0.6)
    
    plt.title("V3.1 vs V4: 6h Ahead RVR Prediction at RWY 09 TDZ")
    plt.xlabel("Sample Index (Latest 1000)")
    plt.ylabel("RVR (Meters)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ROOT / "logs" / "v3_vs_v4_comparison.png")
    print(f"Comparison chart saved to logs/v3_vs_v4_comparison.png")

if __name__ == "__main__":
    run_comparison()
