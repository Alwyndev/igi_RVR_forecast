"""
train_v1_1.py -- Phase 4c: High-Performance V1.1 Refinement

Experimental upgrades:
  1. StandardScaler (Z-score) instead of MinMaxScaler for features/targets.
  2. RVRBiLSTM_V1_1 (Residual connections + Deep FC head).
  3. Increased hidden_size (384) and weight_decay.
"""

import os
import sys
import logging
import json
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

from src.models.model_v1_1 import RVRBiLSTM_V1_1
from src.models.dataset import TARGET_ZONES, TARGET_COLS

# -- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "phase4c_training.log"),
    ],
)
logger = logging.getLogger(__name__)

# -- Config
CONFIG = {
    "project_name":    "IGIA-RVR-BiLSTM",
    "run_name":        "BiLSTM-V1.1-Residual-ZScale",

    # Data
    "parquet_path":    str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet"),
    "scaler_dir":      str(ROOT / "data" / "processed" / "scalers_v1_1"),
    "checkpoint_dir":  str(ROOT / "models"),
    "seq_len":         36,
    "batch_size":      512,
    "num_workers":     2,

    # Architecture
    "hidden_size":     384,
    "num_layers":      2,
    "output_size":     10,
    "dropout":         0.3,

    # Training
    "epochs":          100,
    "lr_max":          1e-3,
    "weight_decay":    1e-3,
    "huber_delta":     50.0,  # 50m delta

    # Early Stopping
    "patience":        20,
}


# ==========================================================================
# Dataset Class
# ==========================================================================
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


def prepare_v1_1_dataloaders():
    logger.info(f"Loading dataset from {CONFIG['parquet_path']}...")
    df = pd.read_parquet(CONFIG["parquet_path"])
    df = df.select_dtypes(include=[np.number])

    target_cols = [c for c in df.columns if c in TARGET_COLS]
    feature_cols = [c for c in df.columns if c not in target_cols]

    logger.info(f"Features: {len(feature_cols)} | Targets: {len(target_cols)}")

    # -- Chronological Split
    train_df = df[df.index.year <= 2023]
    val_df   = df[df.index.year == 2024]
    test_df  = df[df.index.year == 2025]

    logger.info(f"Split sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # -- Feature Scaler (StandardScaler on Train only)
    os.makedirs(CONFIG["scaler_dir"], exist_ok=True)
    scaler_X = StandardScaler()
    scaler_X.fit(train_df[feature_cols])
    joblib.dump(scaler_X, os.path.join(CONFIG["scaler_dir"], "scaler_X.pkl"))

    # -- Target Scaler (StandardScaler on Train only)
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
    """Inverse z-score -> actual metres."""
    p_m = scaler_y.inverse_transform(preds_scaled)
    t_m = scaler_y.inverse_transform(targets_scaled)
    # Clip to physical bounds [0, 10000m]
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
    logger.info(f"Targeting: {device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # -- Data
    train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y = prepare_v1_1_dataloaders()
    input_size = len(feature_cols)

    # -- Model
    model = RVRBiLSTM_V1_1(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=CONFIG["output_size"],
        dropout=CONFIG["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"V1.1 Model Params: {total_params:,}")

    if use_wandb:
        wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)
        wandb.watch(model, log="gradients", log_freq=100)

    # -- Training Rig
    criterion = nn.HuberLoss(delta=1.0) # Delta is in z-score space now, so 1.0 = 1 std dev.
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr_max"],
        steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"],
        pct_start=0.3, anneal_strategy="cos",
    )
    scaler_amp = GradScaler(device='cuda')

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_bilstm_v1_1.pt")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_p, train_t = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler_amp, device, epoch)
        val_loss, val_p, val_t       = evaluate(model, val_loader, criterion, device, desc="Val")

        # MAE in real metres
        train_mae, _, _ = compute_mae_metres(train_p, train_t, scaler_y)
        val_mae, _, _   = compute_mae_metres(val_p, val_t, scaler_y)

        logger.info(
            f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE: {train_mae:.1f}m | Val MAE: {val_mae:.1f}m"
        )

        if use_wandb:
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "train/mae_m": train_mae, "val/mae_m": val_mae, "lr": scheduler.get_last_lr()[0]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_mae": val_mae, "config": CONFIG}, best_ckpt)
            logger.info(f"  [SAVED] Best model (loss={val_loss:.4f}, MAE={val_mae:.1f}m)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    # -- Final Test
    logger.info("\nLoading best V1.1 checkpoint for final evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    test_loss, test_p, test_t = evaluate(model, test_loader, criterion, device, desc="Test")
    test_mae, _, _ = compute_mae_metres(test_p, test_t, scaler_y)

    logger.info("=" * 60)
    logger.info(f"  V1.1 FINAL TEST -> Loss: {test_loss:.4f} | MAE: {test_mae:.1f}m")
    logger.info("=" * 60)

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/mae_m": test_mae})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb)
