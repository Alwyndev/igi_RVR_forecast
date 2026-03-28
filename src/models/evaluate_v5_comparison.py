import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

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
FOG_THRESHOLD = 600

def get_binary_metrics(model, loader, scaler_y):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for X, y in loader:
            p = model(X.to(DEVICE)).cpu().numpy()
            all_p.append(p)
            all_t.append(y.numpy())
    
    p_m = np.clip(scaler_y.inverse_transform(np.concatenate(all_p)), 0, 10000)
    t_m = np.clip(scaler_y.inverse_transform(np.concatenate(all_t)), 0, 10000)
    
    y_true = (t_m < FOG_THRESHOLD).astype(int).flatten()
    y_pred = (p_m < FOG_THRESHOLD).astype(int).flatten()
    
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def run_comparison():
    print("="*60)
    print(f"SAFETY COMPARISON: Standard (V3.1) vs High-Recall (V5)")
    print(f"Threshold: <{FOG_THRESHOLD}m Visibility")
    print("="*60)
    
    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols and df[c].dtype in [np.float32, np.float64, np.int64]]
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
    
    # 2. Evaluate Models
    results = {}
    for name, path in [("V3.1 (Standard)", MODEL_V3_PATH), ("V5 (High-Recall)", MODEL_V5_PATH)]:
        model = RVRAttentionLSTM_V3(input_size=104).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        results[name] = get_binary_metrics(model, loader, scaler_y)
        
    # 3. Output Table
    print("\n" + "="*80)
    print(f"| Model             | Fog Recall (Sensitivity) | Fog Precision (Reliability) | F1-Score |")
    print(f"| :---              | :---:                    | :---:                       | :---:    |")
    for name, m in results.items():
        print(f"| {name:17} | {m['recall']:22.2%} | {m['precision']:25.2%} | {m['f1']:8.4f} |")
    print("="*80)
    print("\nCONCLUSION:")
    v3_rec, v5_rec = results["V3.1 (Standard)"]["recall"], results["V5 (High-Recall)"]["recall"]
    print(f"V5 successfully increased fog-detection recall from {v3_rec:.1%} to {v5_rec:.1%}.")
    print(f"(A {v5_rec/v3_rec:.1f}x improvement in safety sensitivity)")

if __name__ == "__main__":
    run_comparison()
