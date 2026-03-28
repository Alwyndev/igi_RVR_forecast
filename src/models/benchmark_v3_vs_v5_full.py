import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

# -- Config
DATA_PATH = ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"
MODEL_V3_PATH = ROOT / "models" / "best_lstm_v3.pt"
MODEL_V5_PATH = ROOT / "models" / "best_lstm_v5.pt"
SCALER_DIR = ROOT / "data" / "processed" / "scalers_v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOG_THRESHOLD = 600 # Meters

def get_full_metrics(model, loader, scaler_y):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for X, y in loader:
            p = model(X.to(DEVICE)).cpu().numpy()
            all_p.append(p)
            all_t.append(y.numpy())
    
    p_m = np.clip(scaler_y.inverse_transform(np.concatenate(all_p)), 0, 10000)
    t_m = np.clip(scaler_y.inverse_transform(np.concatenate(all_t)), 0, 10000)
    
    # Regression Metrics
    mae = mean_absolute_error(t_m, p_m)
    rmse = np.sqrt(mean_squared_error(t_m, p_m))
    r2 = r2_score(t_m, p_m)
    
    # Threshold Accuracy
    acc100 = np.mean(np.abs(t_m - p_m) <= 100) * 100
    acc200 = np.mean(np.abs(t_m - p_m) <= 200) * 100
    
    # Classification Metrics @ 600m
    y_true = (t_m < FOG_THRESHOLD).astype(int).flatten()
    y_pred = (p_m < FOG_THRESHOLD).astype(int).flatten()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Acc@100m": acc100, "Acc@200m": acc200,
        "Fog Precision": prec * 100, "Fog Recall": rec * 100, "Fog F1": f1
    }

def run_comparison():
    print("="*60)
    print("ALL-ASPECT COMPARISON: V3.1 (Standard) vs V5 (High-Recall)")
    print("="*60)
    
    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    df = df.select_dtypes(include=[np.number]) # CRITICAL: Match train_v3 logic
    
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols]
    test_df = df[df.index.year == 2025].copy()
    
    scaler_X = joblib.load(SCALER_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(SCALER_DIR / "scaler_y.pkl")
    
    X_s = scaler_X.transform(test_df[feature_cols])
    y_s = scaler_y.transform(test_df[target_cols])
    
    from torch.utils.data import Dataset, DataLoader
    class DS(Dataset):
        def __init__(self, x, y): self.x, self.y = torch.FloatTensor(x), torch.FloatTensor(y)
        def __len__(self): return len(self.x) - 36
        def __getitem__(self, idx): return self.x[idx:idx+36], self.y[idx+35]
    
    loader = DataLoader(DS(X_s, y_s), batch_size=256, shuffle=False)
    
    # 2. Results Collection
    metrics = {}
    for name, path in [("V3.1 (Attention LSTM)", MODEL_V3_PATH), ("V5 (High-Recall LSTM)", MODEL_V5_PATH)]:
        model = RVRAttentionLSTM_V3(input_size=104).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        metrics[name] = get_full_metrics(model, loader, scaler_y)
    
    # 3. Formatted Table
    print("\n" + "="*85)
    print(f"| Metric                    | V3.1 (Standard)    | V5 (High-Recall)   | Winner       |")
    print(f"| :---                      | :---:              | :---:              | :---:        |")
    
    m_list = [
        ("Absolute Error (MAE)", "MAE", "V3.1"),
        ("Root Mean Sq Err (RMSE)", "RMSE", "V3.1"),
        ("Coefficient of Det (R2)", "R2", "V3.1"),
        ("Accuracy within 100m", "Acc@100m", "V3.1"),
        ("Accuracy within 200m", "Acc@200m", "V3.1"),
        ("Fog Precision (<600m)", "Fog Precision", "V3.1"),
        ("Fog Recall (Sensitivity)", "Fog Recall", "V5"),
        ("Fog F1-Score (Balanced)", "Fog F1", "V5"),
    ]
    
    for label, key, default_winner in m_list:
        v3_val = metrics["V3.1 (Attention LSTM)"][key]
        v5_val = metrics["V5 (High-Recall LSTM)"][key]
        
        # Decide winner based on metric logic
        if key in ["R2", "Acc@100m", "Acc@200m", "Fog Precision", "Fog Recall", "Fog F1"]:
            win = "V3.1" if v3_val > v5_val else "V5"
        else:
            win = "V3.1" if v3_val < v5_val else "V5"
            
        fmt = "{:8.2f}" if key != "R2" and key != "Fog F1" else "{:8.4f}"
        if "%" in label or "Accuracy" in label or "Fog Precision" in label or "Fog Recall" in label:
            val_v3 = f"{v3_val:7.2f}%"
            val_v5 = f"{v5_val:7.2f}%"
        else:
            val_v3 = fmt.format(v3_val)
            val_v5 = fmt.format(v5_val)
            
        print(f"| {label:25} | {val_v3:18} | {val_v5:18} | {win:12} |")
    
    print("="*85)
    
    print("\nOPERATIONAL PROFILE:")
    print("- V3.1: Recommended for HIGH-RELIABILITY operations (few false alarms, very accurate on clear days).")
    print("- V5: Recommended for SAFETY-FIRST operations (catches more fog onset, but produces more false alerts).")

if __name__ == "__main__":
    run_comparison()
