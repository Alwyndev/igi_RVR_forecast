"""
train_v5.py -- High-Recall Training with Asymmetric Safety Loss (Phase 13)
Encourages the model to be more "pessimistic" during fog events to increase Recall.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from tqdm import tqdm
import joblib

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

# -- Custom Asymmetric Loss
class RVRAsymmetricLoss(nn.Module):
    """
    Penalizes Over-predictions (Pred > Actual) more heavily than Under-predictions
    specifically when conditions are foggy (<600m).
    """
    def __init__(self, fog_threshold_scaled=-3.0, overpred_weight=5.0):
        super().__init__()
        self.fog_threshold = fog_threshold_scaled
        self.overpred_weight = overpred_weight

    def forward(self, pred, actual):
        # Base L1 Loss (MAE)
        loss = torch.abs(pred - actual)
        
        # Dangerous Over-prediction: Actual is Foggy AND Predicted is clearer than Actual
        is_fog = (actual < self.fog_threshold)
        is_overpred = (pred > actual)
        
        # Apply weighting factor (e.g. 5x) to these specific errors
        penalty_mask = (is_fog & is_overpred).float()
        weighted_loss = loss * (1.0 + (self.overpred_weight - 1.0) * penalty_mask)
        
        return weighted_loss.mean()

# -- Config
CONFIG = {
    "project_name":    "IGIA-RVR-BiLSTM",
    "run_name":        "Asymmetric-High-Recall-V5",
    "parquet_path":    str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"),
    "scaler_dir":      str(ROOT / "data" / "processed" / "scalers_v3"), # Re-use v3 scalers
    "checkpoint_dir":  str(ROOT / "models"),
    "seq_len":         36,
    "batch_size":      128,
    "num_workers":     2,
    "hidden_size":     384,
    "num_layers":      3,
    "output_size":     50,
    "dropout":         0.3,
    "epochs":          100,
    "lr_max":          1e-3,
    "weight_decay":    1e-5,
    "patience":        15,
    "fog_weight":      4.5, # Reduced penalty to balance MAE and Recall
}

# (Dataset/Dataloader code similar to train_v3)
class RVRWindowDataset(Dataset):
    def __init__(self, features, targets, seq_len=36):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.features) - self.seq_len
    def __getitem__(self, idx):
        return self.features[idx : idx + self.seq_len], self.targets[idx + self.seq_len - 1]

def prepare_dataloaders():
    df = pd.read_parquet(CONFIG["parquet_path"])
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols and df[c].dtype in [np.float32, np.float64, np.int64]]
    df = df.dropna()
    
    scaler_X = joblib.load(os.path.join(CONFIG["scaler_dir"], "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(CONFIG["scaler_dir"], "scaler_y.pkl"))

    train_df = df[df.index.year <= 2023]
    val_df   = df[df.index.year == 2024]
    test_df  = df[df.index.year == 2025]

    def _scale(d): 
        return scaler_X.transform(d[feature_cols]), scaler_y.transform(d[target_cols])
    
    t_X, t_y = _scale(train_df)
    v_X, v_y = _scale(val_df)
    ts_X, ts_y = _scale(test_df)

    dl_args = {"batch_size": CONFIG["batch_size"], "num_workers": CONFIG["num_workers"], "pin_memory": True}
    train_l = DataLoader(RVRWindowDataset(t_X, t_y), shuffle=True, **dl_args)
    val_l   = DataLoader(RVRWindowDataset(v_X, v_y), shuffle=False, **dl_args)
    test_l  = DataLoader(RVRWindowDataset(ts_X, ts_y), shuffle=False, **dl_args)

    return train_l, val_l, test_l, len(feature_cols), scaler_y

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_l, val_l, test_l, input_size, scaler_y = prepare_dataloaders()

    model = RVRAttentionLSTM_V3(input_size=input_size).to(device)
    
    # Custom safety loss
    criterion = RVRAsymmetricLoss(fog_threshold_scaled=-3.0, overpred_weight=CONFIG["fog_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scaler_amp = GradScaler(device='cuda')

    best_val_loss = float("inf")
    patience = 0
    best_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_lstm_v5.pt")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0
        for X, y in tqdm(train_l, desc=f"Epoch {epoch} [Train]", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(X)
                loss = criterion(preds, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item() * X.size(0)

        model.eval()
        val_loss, all_p, all_t = 0, [], []
        with torch.no_grad():
            for X, y in val_l:
                X, y = X.to(device), y.to(device)
                with autocast(device_type='cuda'):
                    preds = model(X)
                    loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)
                all_p.append(preds.cpu().numpy())
                all_t.append(y.cpu().numpy())
        
        val_loss /= len(val_l.dataset)
        all_p = np.concatenate(all_p)
        all_t = np.concatenate(all_t)
        
        # Classification Accuracy @ 600m
        p_m = np.clip(scaler_y.inverse_transform(all_p), 0, 10000)
        t_m = np.clip(scaler_y.inverse_transform(all_t), 0, 10000)
        y_true = (t_m < 600).astype(int).flatten()
        y_pred = (p_m < 600).astype(int).flatten()
        rec = recall_score(y_true, y_pred, zero_division=0)

        print(f"Epoch {epoch:03d} | Val Loss: {val_loss:.4f} | Recall@600m: {rec:7.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "recall": rec}, best_ckpt)
        else:
            patience += 1
            if patience >= CONFIG["patience"]: break

    print("Training Complete. Final Metrics on 2025 Test Set...")
    model.load_state_dict(torch.load(best_ckpt)["model_state"])
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for X, y in test_l:
            preds = model(X.to(device))
            all_p.append(preds.cpu().numpy())
            all_t.append(y.numpy())
    
    p_m = np.clip(scaler_y.inverse_transform(np.concatenate(all_p)), 0, 10000)
    t_m = np.clip(scaler_y.inverse_transform(np.concatenate(all_t)), 0, 10000)
    y_true = (t_m < 600).astype(int).flatten()
    y_pred = (p_m < 600).astype(int).flatten()
    print(f"FINAL TEST RECALL: {recall_score(y_true, y_pred):.2%}")
    print(f"FINAL TEST PRECISION: {precision_score(y_true, y_pred):.2%}")

if __name__ == "__main__":
    main()
