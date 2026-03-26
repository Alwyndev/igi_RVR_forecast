"""
train_v3.py -- V3 Training Script (Phase 8)

Trains the RVRAttentionLSTM_V3 model to simultaneously predict 
10m, 30m, 1h, 3h, and 6h horizons across 10 runway zones (50 targets).
Strictly uses an alphabetical target sorting standard to perfectly match external benchmark parameters.
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
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import joblib
import wandb

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3
from src.data.runway_config import CONSOLIDATED_ZONES

# -- Target Definition
# To ensure perfect alignment with the external benchmark, we sort the targets alphabetically
HORIZONS = ["10m", "30m", "1h", "3h", "6h"]

# -- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "phase8_v3_training.log"),
    ],
)
logger = logging.getLogger(__name__)

# -- Config
CONFIG = {
    "project_name":    "IGIA-RVR-BiLSTM",
    "run_name":        "Residual-Attention-LSTM-V3",

    # Data
    "parquet_path":    str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"),
    "scaler_dir":      str(ROOT / "data" / "processed" / "scalers_v3"),
    "checkpoint_dir":  str(ROOT / "models"),
    "seq_len":         36,
    "batch_size":      128,
    "num_workers":     2,

    # Architecture
    "hidden_size":     384,
    "num_layers":      3,
    "output_size":     50,
    "dropout":         0.3,

    # Training
    "epochs":          100,
    "lr_max":          1e-3,
    "weight_decay":    1e-3,

    # Early Stopping
    "patience":        15,
}

class RVRWindowDataset(Dataset):
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


def prepare_multi_dataloaders():
    logger.info(f"Loading {CONFIG['parquet_path']}...")
    df = pd.read_parquet(CONFIG["parquet_path"])
    df = df.select_dtypes(include=[np.number])

    # Enforce alphabetical sorting of targets to match external model EXACTLY
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    if len(target_cols) != 50:
        raise ValueError(f"Expected 50 targets, found {len(target_cols)}")

    # Exclude all target columns from features (to prevent leakage)
    feature_cols = [c for c in df.columns if c not in target_cols]

    logger.info(f"Features: {len(feature_cols)} | Targets: {len(target_cols)} (Alphabetical Sorting)")

    # -- Drop any remaining NaNs or Infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    logger.info(f"Dataset shape after purging NaNs/Infs: {df.shape}")

    # -- Split
    train_df = df[df.index.year <= 2023]
    val_df   = df[df.index.year == 2024]
    test_df  = df[df.index.year == 2025]

    logger.info(f"Split sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # -- Scale
    os.makedirs(CONFIG["scaler_dir"], exist_ok=True)
    scaler_X = StandardScaler()
    scaler_X.fit(train_df[feature_cols])
    scaler_X.feature_names_in_ = np.array(feature_cols)
    joblib.dump(scaler_X, os.path.join(CONFIG["scaler_dir"], "scaler_X.pkl"))

    scaler_y = StandardScaler()
    scaler_y.fit(train_df[target_cols])
    joblib.dump(scaler_y, os.path.join(CONFIG["scaler_dir"], "scaler_y.pkl"))

    def _scale_split(df_split):
        X = scaler_X.transform(df_split[feature_cols])
        y = scaler_y.transform(df_split[target_cols])
        return X, y

    train_X, train_y = _scale_split(train_df)
    val_X,   val_y   = _scale_split(val_df)
    test_X,  test_y  = _scale_split(test_df)

    # -- Datasets
    train_ds = RVRWindowDataset(train_X, train_y, CONFIG["seq_len"])
    val_ds   = RVRWindowDataset(val_X,   val_y,   CONFIG["seq_len"])
    test_ds  = RVRWindowDataset(test_X,  test_y,  CONFIG["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y


def compute_mae_metres(preds_scaled, targets_scaled, scaler_y):
    p_m = scaler_y.inverse_transform(preds_scaled)
    t_m = scaler_y.inverse_transform(targets_scaled)
    p_m = np.clip(p_m, 0, 10000)
    t_m = np.clip(t_m, 0, 10000)
    return mean_absolute_error(t_m, p_m), p_m, t_m


def train_one_epoch_simple(model, loader, criterion, optimizer, scaler_amp, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for X, y in pbar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            preds = model(X)
            loss  = criterion(preds, y)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        lv = loss.item()
        total_loss += lv * X.size(0)
        pbar.set_postfix({"loss": f"{lv:.4f}"})
        
        if len(all_preds) < 5:
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(all_preds), np.concatenate(all_targets)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for X, y in pbar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type='cuda'):
            preds = model(X)
            loss  = criterion(preds, y)
        total_loss += loss.item() * X.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(all_preds), np.concatenate(all_targets)


def main(use_wandb=True):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Targeting: {device}")

    # -- Data
    train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y = prepare_multi_dataloaders()

    # -- Model
    model = RVRAttentionLSTM_V3(
        input_size=len(feature_cols),
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=CONFIG["output_size"],
        dropout=CONFIG["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Multi-Horizon V3 Model Params: {total_params:,}")

    if use_wandb:
        wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_max"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    scaler_amp = GradScaler(device='cuda')

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_lstm_v3.pt")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_p, train_t = train_one_epoch_simple(model, train_loader, criterion, optimizer, scaler_amp, device, epoch)
        val_loss, val_p, val_t       = evaluate(model, val_loader, criterion, device, desc="Val")

        # Update scheduler
        scheduler.step(val_loss)

        # Train MAE is just an estimate based on last few batches to save RAM
        train_mae, _, _ = compute_mae_metres(train_p, train_t, scaler_y)
        val_mae, val_p_m, val_t_m = compute_mae_metres(val_p, val_t, scaler_y)

        logger.info(
            f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE (sample): {train_mae:.1f}m | Val MAE (Overall): {val_mae:.1f}m"
        )

        log_dict = {"train/loss": train_loss, "val/loss": val_loss, "train/mae_m_sample": train_mae, "val/mae_m": val_mae}
        
        if use_wandb:
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_mae": val_mae, "config": CONFIG}, best_ckpt)
            logger.info(f"  [SAVED] Best V3 model (loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    if use_wandb: wandb.finish()
    
    # Final Test Set Evaluation
    logger.info("Running Best Model on Unseen 2025 Test Set...")
    best_state = torch.load(best_ckpt)["model_state"]
    model.load_state_dict(best_state)
    
    test_loss, test_p, test_t = evaluate(model, test_loader, criterion, device, desc="Test")
    test_mae, _, _ = compute_mae_metres(test_p, test_t, scaler_y)
    logger.info(f"FINAL TEST MAE (2025 Unseen): {test_mae:.2f}m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb)
