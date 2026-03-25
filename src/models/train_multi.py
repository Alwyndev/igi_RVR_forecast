"""
train_multi.py -- Multi-Horizon Training Script (Phase 6)

Trains the RVRBiLSTM_Multi model to simultaneously predict 
10m, 30m, 1h, 3h, and 6h horizons across 10 runway zones (50 targets).
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

from src.models.model_multi import RVRBiLSTM_Multi
from src.data.runway_config import CONSOLIDATED_ZONES

# -- Target Definition (Order matters!)
TARGET_ZONES = CONSOLIDATED_ZONES
HORIZONS = ["10m", "30m", "1h", "3h", "6h"]
TARGET_COLS_MULTI = [f"target_{z}_rvr_actual_mean_{h}" for z in TARGET_ZONES for h in HORIZONS]

# -- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "phase6_multi_horizon.log"),
    ],
)
logger = logging.getLogger(__name__)

# -- Config
CONFIG = {
    "project_name":    "IGIA-RVR-BiLSTM",
    "run_name":        "BiLSTM-MultiHorizon",

    # Data
    "parquet_path":    str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"),
    "scaler_dir":      str(ROOT / "data" / "processed" / "scalers_multi"),
    "checkpoint_dir":  str(ROOT / "models"),
    "seq_len":         36,
    "batch_size":      512,
    "num_workers":     2,

    # Architecture
    "hidden_size":     384,
    "num_layers":      2,
    "output_size":     50, # 10 zones * 5 horizons
    "dropout":         0.3,

    # Training
    "epochs":          100,
    "lr_max":          1e-3,
    "weight_decay":    1e-3,

    # Early Stopping
    "patience":        20,
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

    # Ensure our target cols exist
    missing = [c for c in TARGET_COLS_MULTI if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    target_cols = TARGET_COLS_MULTI
    
    # Exclude all target columns from features (to prevent leakage)
    all_targets_in_dataset = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in all_targets_in_dataset]

    logger.info(f"Features: {len(feature_cols)} | Targets: {len(target_cols)}")

    # -- Drop any remaining NaNs or Infs (e.g., from temporal shifts or divides)
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

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y


def compute_mae_metres(preds_scaled, targets_scaled, scaler_y):
    p_m = scaler_y.inverse_transform(preds_scaled)
    t_m = scaler_y.inverse_transform(targets_scaled)
    p_m = np.clip(p_m, 0, 10000)
    t_m = np.clip(t_m, 0, 10000)
    return mean_absolute_error(t_m, p_m), p_m, t_m


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler_amp, device, epoch):
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
        scheduler.step()

        lv = loss.item()
        total_loss += lv * X.size(0)
        pbar.set_postfix({"loss": f"{lv:.4f}"})
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
    model = RVRBiLSTM_Multi(
        input_size=len(feature_cols),
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=CONFIG["output_size"],
        dropout=CONFIG["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Multi-Horizon Model Params: {total_params:,}")

    if use_wandb:
        wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr_max"],
        steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"],
        pct_start=0.3, anneal_strategy="cos",
    )
    scaler_amp = GradScaler(device='cuda')

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_bilstm_multi.pt")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_p, train_t = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler_amp, device, epoch)
        val_loss, val_p, val_t       = evaluate(model, val_loader, criterion, device, desc="Val")

        train_mae, _, _ = compute_mae_metres(train_p, train_t, scaler_y)
        val_mae, val_p_m, val_t_m = compute_mae_metres(val_p, val_t, scaler_y)

        logger.info(
            f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE (Overall): {train_mae:.1f}m | Val MAE (Overall): {val_mae:.1f}m"
        )

        log_dict = {"train/loss": train_loss, "val/loss": val_loss, "train/mae_m": train_mae, "val/mae_m": val_mae}
        
        # Log specific horizons (Optional depth for W&B)
        # Horizons repeat every 5 steps for a given zone. Let's average by horizon.
        # Format: zone0_h0, zone0_h1 ... zone0_h4, zone1_h0 ...
        # If we reshape back to (batch, 10, 5), we can get MAE per horizon
        p_m_reshaped = val_p_m.reshape(-1, 10, 5)
        t_m_reshaped = val_t_m.reshape(-1, 10, 5)
        for h_idx, h_label in enumerate(HORIZONS):
            h_mae = mean_absolute_error(t_m_reshaped[:, :, h_idx].flatten(), p_m_reshaped[:, :, h_idx].flatten())
            log_dict[f"val/mae_{h_label}"] = h_mae

        if use_wandb:
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_mae": val_mae, "config": CONFIG}, best_ckpt)
            logger.info(f"  [SAVED] Best Multi-Horizon model (loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    if use_wandb: wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb)
