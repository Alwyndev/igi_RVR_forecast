import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys
from tqdm import tqdm
from pathlib import Path

# -- Local Imports
ROOT = Path("c:/Users/alwyn/OneDrive/Desktop/IGI_Antigravity")
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

# -- Config
DATA_PATH = ROOT / "data/processed/igia_rvr_training_dataset_multi.parquet"
SCALER_DIR = ROOT / "data/processed/scalers_v3"
MODEL_PATH = ROOT / "models/best_lstm_v3.pt"
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

def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.select_dtypes(include=[np.number])

    # Enforce alphabetical sorting
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    # Drop NaNs/Infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    # Split
    val_df   = df[df.index.year == 2024]
    test_df  = df[df.index.year == 2025]
    
    # Scalers
    scaler_X = joblib.load(os.path.join(SCALER_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(SCALER_DIR, "scaler_y.pkl"))
    
    def _scale_split(df_split):
        X = scaler_X.transform(df_split[feature_cols])
        y = scaler_y.transform(df_split[target_cols])
        return X, y

    val_X,   val_y   = _scale_split(val_df)
    test_X,  test_y  = _scale_split(test_df)
    
    # Datasets
    val_ds   = RVRWindowDataset(val_X,   val_y, seq_len=36)
    test_ds  = RVRWindowDataset(test_X,  test_y, seq_len=36)
    
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    
    return val_loader, test_loader, scaler_y, len(feature_cols)

def main():
    print(f"Targeting: {DEVICE}")
    print(f"Loading V3.1 model from {MODEL_PATH}...")
    val_loader, test_loader, scaler_y, input_size = load_data()
    
    # Reconstruct architecture
    model = RVRAttentionLSTM_V3(
        input_size=input_size,
        hidden_size=384,
        num_layers=3,
        output_size=50,
        dropout=0.3
    ).to(DEVICE)
    
    # Load state dict correctly
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Accuracy Logic
    def accuracy_at_threshold(all_targets_orig, all_preds_orig, threshold=200):
        errors = np.abs(all_targets_orig - all_preds_orig)
        accuracy = np.mean(errors <= threshold) * 100
        return accuracy

    print("Running inference on 2024 (Val)...")
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Val"):
            preds = model(X.to(DEVICE)).cpu().numpy()
            val_preds.append(preds)
            val_targets.append(y.numpy())

    val_preds_orig = scaler_y.inverse_transform(np.concatenate(val_preds))
    val_targets_orig = scaler_y.inverse_transform(np.concatenate(val_targets))

    print("Running inference on 2025 (Test)...")
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Test"):
            preds = model(X.to(DEVICE)).cpu().numpy()
            test_preds.append(preds)
            test_targets.append(y.numpy())

    test_preds_orig = scaler_y.inverse_transform(np.concatenate(test_preds))
    test_targets_orig = scaler_y.inverse_transform(np.concatenate(test_targets))

    print("\n" + "="*50)
    print("✓ Accuracy@Xm (% predictions within X meters of actual):")
    for thresh in [100, 150, 200, 250, 300]:
        val_acc = accuracy_at_threshold(val_targets_orig, val_preds_orig, thresh)
        test_acc = accuracy_at_threshold(test_targets_orig, test_preds_orig, thresh)
        print(f"  Accuracy@{thresh:3d}m: | 2024: {val_acc:6.2f}% | 2025: {test_acc:6.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
