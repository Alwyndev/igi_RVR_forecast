import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

# -- Config
DATA_PATH = ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"
MODEL_V3_PATH = ROOT / "models" / "best_lstm_v3.pt"
SCALER_V3_DIR = ROOT / "data" / "processed" / "scalers_v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOG_THRESHOLD = 600 # Meters (CAT-II Transition)

def run_classification_eval():
    print("="*60)
    print(f"V3.1 CLASSIFICATION EVALUATION (Threshold: {FOG_THRESHOLD}m)")
    print("="*60)
    
    # 1. Load Data (2025 Test Set)
    df = pd.read_parquet(DATA_PATH)
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols and df[c].dtype in [np.float32, np.float64, np.int64]]
    test_df = df[df.index.year == 2025].copy()
    
    # 2. Load Model & Scalers
    scaler_X = joblib.load(SCALER_V3_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(SCALER_V3_DIR / "scaler_y.pkl")
    
    model = RVRAttentionLSTM_V3(input_size=104).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_V3_PATH, map_location=DEVICE)["model_state"])
    model.eval()
    
    X_scaled = scaler_X.transform(test_df[feature_cols])
    y_scaled = scaler_y.transform(test_df[target_cols])
    
    # 3. Inference
    all_preds, all_actuals = [], []
    for i in tqdm(range(len(test_df) - 36), desc="Running Inference"):
        win = torch.FloatTensor(X_scaled[i : i+36]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = model(win).cpu().numpy()
        all_preds.append(p)
        all_actuals.append(y_scaled[i + 35])
        
    all_preds = np.concatenate(all_preds)
    all_actuals = np.array(all_actuals)
    
    # Inverse Transform to Meters
    preds_m = np.clip(scaler_y.inverse_transform(all_preds), 0, 10000)
    act_m = np.clip(scaler_y.inverse_transform(all_actuals), 0, 10000)
    
    # 4. Binary Conversion (Fog < 600m)
    # 1 = Fog Event (Hazard), 0 = Clear
    y_true = (act_m < FOG_THRESHOLD).astype(int).flatten()
    y_pred = (preds_m < FOG_THRESHOLD).astype(int).flatten()
    
    # 5. Metrics
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 6. Report
    print("\n" + "="*60)
    print("FOG DETECTION ACCURACY (Operational Performance)")
    print("-" * 60)
    print(f"| Metric            | Value (Overall) |")
    print(f"| :---              | :---:           |")
    print(f"| Fog Precision     | {prec:7.2%}        |  (Of predicted fog, how many were real?)")
    print(f"| Fog Recall        | {rec:7.2%}        |  (Of real fog events, how many were caught?)")
    print(f"| F1-Score          | {f1:7.4f}         |")
    print("-" * 60)
    print(f"Total True Positives (Fog Caught): {tp}")
    print(f"Total False Negatives (Fog Missed): {fn} (Critical Error)")
    print(f"Total False Positives (False Alarm): {fp}")
    print(f"Total True Negatives (Correct Clear): {tn}")
    print("="*60)
    
    # Horizon Breakdown (Show 6h ahead recall)
    print("\nRECALL BY HORIZON (RWY 09 TDZ):")
    horizons = ["10m", "30m", "1h", "3h", "6h"]
    for h_idx, h_name in enumerate(horizons):
        h_true = (act_m[:, h_idx] < FOG_THRESHOLD).astype(int)
        h_pred = (preds_m[:, h_idx] < FOG_THRESHOLD).astype(int)
        h_rec = recall_score(h_true, h_pred)
        print(f"  - Horizon {h_name:4}: {h_rec:7.2%}")

if __name__ == "__main__":
    run_classification_eval()
