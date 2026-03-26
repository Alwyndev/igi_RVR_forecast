import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- Local Imports
ROOT = Path("c:/Users/alwyn/OneDrive/Desktop/IGI_Antigravity")
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

# -- Config
DATA_PATH = ROOT / "data/processed/igia_rvr_training_dataset_multi.parquet"
MODEL_CHAMPION = ROOT / "models/best_lstm_v3.pt"
MODEL_SEASONAL = ROOT / "models/best_lstm_v3_seasonal.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RVRWindowDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, seq_len=36):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return x, y

def get_eval_data(scaler_dir, months=[12, 1, 2]):
    df = pd.read_parquet(DATA_PATH)
    df = df.select_dtypes(include=[np.number])

    # Enforce alphabetical sorting
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    # Filter for the FAIR Dec-Feb Window only (2024-25)
    test_df = df[(df.index.year >= 2024) & (df.index.month.isin(months))].copy()
    
    # Scalers
    scaler_X = joblib.load(os.path.join(scaler_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))
    
    X = scaler_X.transform(test_df[feature_cols])
    y = scaler_y.transform(test_df[target_cols])
    
    ds = RVRWindowDataset(X, y, seq_len=36)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    
    return loader, scaler_y, len(feature_cols)

def evaluate_model(model_path, is_seasonal=False):
    scaler_dir = ROOT / "data/processed/scalers_v3"
    if is_seasonal:
        scaler_dir = str(scaler_dir) + "_seasonal"
    
    loader, scaler_y, input_size = get_eval_data(scaler_dir)
    
    model = RVRAttentionLSTM_V3(input_size=input_size).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Eval"):
            preds = model(X.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
            
    p_orig = scaler_y.inverse_transform(np.concatenate(all_preds))
    t_orig = scaler_y.inverse_transform(np.concatenate(all_targets))
    
    # Clip to physical bounds
    p_orig = np.clip(p_orig, 0, 10000)
    
    mae = mean_absolute_error(t_orig, p_orig)
    rmse = np.sqrt(mean_squared_error(t_orig, p_orig))
    r2 = r2_score(t_orig, p_orig)
    
    return mae, rmse, r2

def main():
    print("="*50)
    print("FAIR BENCHMARK: Dec 2024 - Feb 2025 ONLY")
    print("="*50)
    
    print("\n[1/2] Evaluating Full-Year Champion...")
    mae_c, rmse_c, r2_c = evaluate_model(MODEL_CHAMPION, is_seasonal=False)
    
    print("\n[2/2] Evaluating Seasonal Oct-Mar Specialist...")
    mae_s, rmse_s, r2_s = evaluate_model(MODEL_SEASONAL, is_seasonal=True)
    
    print("\n" + "="*50)
    print("FINAL COMPARISON (Dec-Feb Hard Window)")
    print(f"| Metric | Champion (Full) | Seasonal (Oct-Mar) | Delta |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| MAE | {mae_c:7.2f}m | {mae_s:7.2f}m | {mae_s-mae_c:7.2f}m |")
    print(f"| RMSE | {rmse_c:7.2f}m | {rmse_s:7.2f}m | {rmse_s-rmse_c:7.2f}m |")
    print(f"| R2 | {r2_c:7.4f} | {r2_s:7.4f} | {r2_s-r2_c:7.4f} |")
    print("="*50)

if __name__ == "__main__":
    main()
