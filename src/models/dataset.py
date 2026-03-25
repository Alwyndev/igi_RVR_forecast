"""
dataset.py — PyTorch Dataset & DataLoader for IGIA RVR BiLSTM

Implements:
  - Chronological split: Train=2019-2023, Val=2024, Test=2025
  - MinMaxScaler fitted ONLY on Training set (no leakage)
  - Sliding window sequences → (seq_len, features) tensors
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Canonical zone ordering (must match model output indices)
# ============================================================
TARGET_ZONES = [
    "09_TDZ", "27_TDZ", "10_TDZ", "28_TDZ", "MID_2810",
    "11_TDZ", "11_BEG", "29_TDZ", "29_BEG", "MID_2911",
]
TARGET_COLS = [f"target_{z}_rvr_actual_mean_6h" for z in TARGET_ZONES]


class RVRWindowDataset(Dataset):
    """
    Sliding-window dataset for BiLSTM training.

    Each sample is:
      X : (seq_len, n_features) — the past `seq_len` 10-minute steps
      y : (10,)                 — RVR prediction 6 hours ahead, all zones
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 36):
        """
        Args:
            features : 2D array (T, F) — scaled feature matrix
            targets  : 2D array (T, 10) — scaled target matrix
            seq_len  : sliding window length (default: 36 = 6 hours at 10-min intervals)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]        # (seq_len, F)
        y = self.targets[idx + self.seq_len - 1]           # (10,)
        return x, y


def prepare_dataloaders(
    parquet_path: str,
    scaler_dir: str,
    seq_len: int = 36,
    batch_size: int = 512,
    num_workers: int = 4,
):
    """
    Load, split, scale, and wrap data into DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y
    """
    logger.info(f"Loading finalized dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Separate features vs targets
    target_cols = [c for c in df.columns if c in TARGET_COLS]
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    logger.info(f"Features: {len(feature_cols)} | Targets: {len(target_cols)}")

    # ── Chronological Split ──────────────────────────────────────────────────
    train_mask = df.index.year <= 2023
    val_mask   = df.index.year == 2024
    test_mask  = df.index.year == 2025

    train_df = df[train_mask]
    val_df   = df[val_mask]
    test_df  = df[test_mask]

    logger.info(
        f"Split sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    # ── Fit Scaler ONLY on Train ─────────────────────────────────────────────
    os.makedirs(scaler_dir, exist_ok=True)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(train_df[feature_cols])

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(train_df[target_cols])

    # Persist scalers for inference
    joblib.dump(scaler_X, os.path.join(scaler_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(scaler_dir, "scaler_y.pkl"))
    logger.info(f"Scalers saved to {scaler_dir}")

    def _scale(df_split):
        X = scaler_X.transform(df_split[feature_cols])
        y = scaler_y.transform(df_split[target_cols])
        return X, y

    train_X, train_y = _scale(train_df)
    val_X,   val_y   = _scale(val_df)
    test_X,  test_y  = _scale(test_df)

    # ── PyTorch Datasets ─────────────────────────────────────────────────────
    train_ds = RVRWindowDataset(train_X, train_y, seq_len)
    val_ds   = RVRWindowDataset(val_X,   val_y,   seq_len)
    test_ds  = RVRWindowDataset(test_X,  test_y,  seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y
