"""
train_v2.py -- Phase 4b: Optimized Training with Log-Transform + Attention

Key changes from train.py (V1):
  1. Targets are Log1p-transformed: model predicts log1p(RVR), unscaled back to metres for MAE.
  2. Uses RVRBiLSTMAttention (model_v2) with Self-Attention and larger hidden dim.
  3. Keeps all V1 strengths: AMP, AdamW, OneCycleLR, W&B, Early Stopping.
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import joblib
import wandb

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v2 import RVRBiLSTMAttention
from src.models.dataset import TARGET_ZONES, TARGET_COLS

# -- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "phase4b_training.log"),
    ],
)
logger = logging.getLogger(__name__)

# -- Config
CONFIG = {
    "project_name":    "IGIA-RVR-BiLSTM",
    "run_name":        "BiLSTM-Attention-v2-LogRVR",

    # Data
    "parquet_path":    str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet"),
    "scaler_dir":      str(ROOT / "data" / "processed" / "scalers_v2"),
    "checkpoint_dir":  str(ROOT / "models"),
    "seq_len":         36,
    "batch_size":      256,     # Smaller batch for larger model
    "num_workers":     2,

    # Architecture
    "hidden_size":     256,     # Restored to 256 for stable linear targets
    "num_layers":      2,
    "output_size":     10,
    "num_heads":       4,
    "dropout":         0.3,

    # Training
    "epochs":          80,
    "lr_max":          1e-3,
    "weight_decay":    5e-4,    # Balanced regularisation
    "huber_delta":     50.0,    # Restored to 50m (linear delta)

    # Early Stopping
    "patience":        15,
}


# ==========================================================================
# Dataset with Log-Transform
# ==========================================================================
class RVRWindowDatasetLog(Dataset):
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


def prepare_v2_dataloaders():
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

    logger.info(f"Split -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # -- Feature Scaler (MinMax on Train only)
    os.makedirs(CONFIG["scaler_dir"], exist_ok=True)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(train_df[feature_cols])
    joblib.dump(scaler_X, os.path.join(CONFIG["scaler_dir"], "scaler_X.pkl"))

    # -- Target Scaler (MinMax on Train only)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(train_df[target_cols])
    joblib.dump(scaler_y, os.path.join(CONFIG["scaler_dir"], "scaler_y.pkl"))

    def _scale_features(df_split):
        return scaler_X.transform(df_split[feature_cols])

    train_X = _scale_features(train_df)
    val_X   = _scale_features(val_df)
    test_X  = _scale_features(test_df)

    train_y = scaler_y.transform(train_df[target_cols])
    val_y   = scaler_y.transform(val_df[target_cols])
    test_y  = scaler_y.transform(test_df[target_cols])

    # -- Datasets
    seq = CONFIG["seq_len"]
    bs  = CONFIG["batch_size"]
    nw  = CONFIG["num_workers"]

    train_ds = RVRWindowDatasetLog(train_X, train_y, seq)
    val_ds   = RVRWindowDatasetLog(val_X,   val_y,   seq)
    test_ds  = RVRWindowDatasetLog(test_X,  test_y,  seq)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))

    return train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y


def compute_mae_metres(preds_scaled, targets_scaled, scaler_y):
    """Inverse: scaled -> real metres."""
    preds_m   = scaler_y.inverse_transform(preds_scaled)
    targets_m = scaler_y.inverse_transform(targets_scaled)
    # Clip to physical range
    preds_m   = np.clip(preds_m, 0, 10000)
    targets_m = np.clip(targets_m, 0, 10000)
    return mean_absolute_error(targets_m, preds_m), preds_m, targets_m


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
    os.makedirs(ROOT / "logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # -- Data
    train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y = prepare_v2_dataloaders()
    input_size = len(feature_cols)

    # -- Training Plan
    plan = {
        "Model":        "BiLSTM-Attention V2",
        "Target":       "log1p(RVR) scaled",
        "Input":        input_size,
        "Hidden":       CONFIG["hidden_size"],
        "Attention":    f"{CONFIG['num_heads']} heads",
        "Train":        f"{len(train_loader.dataset):,}",
        "Val":          f"{len(val_loader.dataset):,}",
        "Test":         f"{len(test_loader.dataset):,}",
        "Epochs":       CONFIG["epochs"],
        "LR":           CONFIG["lr_max"],
        "Loss":         f"Huber (delta={CONFIG['huber_delta']})",
    }
    logger.info("\n" + "=" * 55)
    logger.info("    IGIA RVR BiLSTM-Attention V2 -- TRAINING PLAN")
    logger.info("=" * 55)
    for k, v in plan.items():
        logger.info(f"  {k:<18}: {v}")
    logger.info("=" * 55 + "\n")

    with open(ROOT / "logs" / "phase4b_training_plan.json", "w") as f:
        json.dump(plan, f, indent=4)

    # -- W&B
    if use_wandb:
        wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)

    # -- Model
    model = RVRBiLSTMAttention(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=CONFIG["output_size"],
        num_heads=CONFIG["num_heads"],
        dropout=CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,}")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    # -- Training Setup
    criterion  = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer  = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr_max"],
        steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"],
        pct_start=0.3, anneal_strategy="cos",
    )
    scaler_amp = GradScaler(device='cuda')

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_bilstm_v2.pt")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler_amp, device, epoch
        )
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device, desc="Val")

        train_mae, _, _ = compute_mae_metres(train_preds, train_targets, scaler_y)
        val_mae, _, _   = compute_mae_metres(val_preds, val_targets, scaler_y)

        logger.info(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE: {train_mae:.1f}m | Val MAE: {val_mae:.1f}m"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch, "train/loss": train_loss, "val/loss": val_loss,
                "train/mae_m": train_mae, "val/mae_m": val_mae,
                "lr": scheduler.get_last_lr()[0],
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "val_loss": val_loss,
                "val_mae_m": val_mae, "config": CONFIG,
            }, best_ckpt)
            logger.info(f"  [SAVED] Best model (val_loss={val_loss:.4f}, val_mae={val_mae:.1f}m)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    # -- Final Test
    logger.info("\nLoading best V2 checkpoint for final test...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device, desc="Test")
    test_mae, _, _ = compute_mae_metres(test_preds, test_targets, scaler_y)

    logger.info("=" * 55)
    logger.info(f"  FINAL TEST V2 -> Loss: {test_loss:.4f} | MAE: {test_mae:.1f}m")
    logger.info("=" * 55)

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/mae_m": test_mae})
        wandb.finish()

    results = {
        "model": "BiLSTM-Attention V2 (Log-Transform)",
        "best_epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"],
        "val_mae_m": ckpt["val_mae_m"], "test_loss": test_loss, "test_mae_m": test_mae,
    }
    with open(ROOT / "logs" / "phase4b_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Results saved to logs/phase4b_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb)
