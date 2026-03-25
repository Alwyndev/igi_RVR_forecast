"""
train.py — Phase 4 Training Loop for IGIA RVR BiLSTM

Implements:
  - Chronological Train(2019-23) / Val(2024) / Test(2025) split
  - AdamW optimiser with OneCycleLR scheduler
  - Huber Loss (robust to outliers in fog/non-fog distribution)
  - AMP (torch.cuda.amp) for Tensor Core acceleration on RTX 5070 Ti
  - W&B experiment tracking
  - Best model checkpointing
  - Epoch-level MAE in meters (unscaled)
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

import wandb

# ── Local Imports ────────────────────────────────────────────────────────────
# Allow running from project root: python -m src.models.train
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model import RVRBiLSTM
from src.models.dataset import prepare_dataloaders

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "phase4_training.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "project_name":    "IGIA-RVR-BiLSTM",
    "run_name":        "BiLSTM-v1-Huber",
    
    # Data
    "parquet_path":    str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet"),
    "scaler_dir":      str(ROOT / "data" / "processed" / "scalers"),
    "checkpoint_dir":  str(ROOT / "models"),
    "seq_len":         36,        # 6h at 10-min intervals
    "batch_size":      512,
    "num_workers":     4,

    # Architecture
    "hidden_size":     256,
    "num_layers":      2,
    "output_size":     10,
    "dropout":         0.2,

    # Training
    "epochs":          50,
    "lr_max":          1e-3,
    "weight_decay":    1e-4,
    "huber_delta":     50.0,      # meters — beyond this Huber switches to linear

    # Early Stopping
    "patience":        10,
}


def compute_mae_meters(preds_scaled, targets_scaled, scaler_y):
    """Inverse-transform and return MAE in real metres."""
    preds_m   = scaler_y.inverse_transform(preds_scaled)
    targets_m = scaler_y.inverse_transform(targets_scaled)
    return mean_absolute_error(targets_m, preds_m)


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
        # Gradient clipping for stability
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val * X.size(0)
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return avg_loss, all_preds, all_targets


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

    avg_loss = total_loss / len(loader.dataset)
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return avg_loss, all_preds, all_targets


def main(use_wandb: bool = True):
    # ── Dirs & Device ─────────────────────────────────────────────────────────
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(ROOT / "logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_y = prepare_dataloaders(
        parquet_path = CONFIG["parquet_path"],
        scaler_dir   = CONFIG["scaler_dir"],
        seq_len      = CONFIG["seq_len"],
        batch_size   = CONFIG["batch_size"],
        num_workers  = CONFIG["num_workers"],
    )
    input_size = len(feature_cols)

    # ── Training Plan ─────────────────────────────────────────────────────────
    plan = {
        "Device":           str(device),
        "GPU":              torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "Input Features":   input_size,
        "Sequence Length":  f"{CONFIG['seq_len']} steps (6h)",
        "Train Samples":    f"{len(train_loader.dataset):,}",
        "Val Samples":      f"{len(val_loader.dataset):,}",
        "Test Samples":     f"{len(test_loader.dataset):,}",
        "Architecture":     f"BiLSTM × {CONFIG['num_layers']} | hidden={CONFIG['hidden_size']} | dropout={CONFIG['dropout']}",
        "Outputs":          f"{CONFIG['output_size']} RVR zones",
        "Batch Size":       CONFIG["batch_size"],
        "Epochs":           CONFIG["epochs"],
        "Optimiser":        "AdamW",
        "LR (max)":         CONFIG["lr_max"],
        "Scheduler":        "OneCycleLR",
        "Loss Function":    f"HuberLoss (delta={CONFIG['huber_delta']}m)",
        "AMP":              "Enabled (RTX 5070 Ti Tensor Cores)",
        "W&B Tracking":     "Enabled" if use_wandb else "Disabled",
    }

    logger.info("\n" + "=" * 55)
    logger.info("          IGIA RVR BiLSTM — TRAINING PLAN")
    logger.info("=" * 55)
    for k, v in plan.items():
        logger.info(f"  {k:<22}: {v}")
    logger.info("=" * 55 + "\n")

    # Save plan to logs
    with open(ROOT / "logs" / "phase4_training_plan.json", "w") as f:
        json.dump(plan, f, indent=4)

    # ── W&B Init ──────────────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project = CONFIG["project_name"],
            name    = CONFIG["run_name"],
            config  = CONFIG,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RVRBiLSTM(
        input_size  = input_size,
        hidden_size = CONFIG["hidden_size"],
        num_layers  = CONFIG["num_layers"],
        output_size = CONFIG["output_size"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialised with {total_params:,} trainable parameters.")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    # ── Criterion / Optimiser / Scheduler / AMP ───────────────────────────────
    criterion  = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer  = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = CONFIG["lr_max"],
        steps_per_epoch = len(train_loader),
        epochs        = CONFIG["epochs"],
        pct_start     = 0.3,
        anneal_strategy = "cos",
    )
    scaler_amp = GradScaler(device='cuda')

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_bilstm.pt")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler_amp, device, epoch
        )
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device, desc="Val")

        # MAE in metres (unscaled)
        train_mae = compute_mae_meters(train_preds, train_targets, scaler_y)
        val_mae   = compute_mae_meters(val_preds, val_targets, scaler_y)

        logger.info(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE: {train_mae:.1f}m | Val MAE: {val_mae:.1f}m"
        )

        if use_wandb:
            wandb.log({
                "epoch":      epoch,
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "train/mae_m": train_mae,
                "val/mae_m":   val_mae,
                "lr":         scheduler.get_last_lr()[0],
            })

        # ── Early Stopping & Checkpointing ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_loss":     val_loss,
                "val_mae_m":    val_mae,
                "config":       CONFIG,
            }, best_ckpt)
            logger.info(f"  [SAVED] New best model saved (val_loss={val_loss:.4f}, val_mae={val_mae:.1f}m)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device, desc="Test")
    test_mae = compute_mae_meters(test_preds, test_targets, scaler_y)

    logger.info("=" * 55)
    logger.info(f"  FINAL TEST -> Loss: {test_loss:.4f} | MAE: {test_mae:.1f}m")
    logger.info("=" * 55)

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/mae_m": test_mae})
        wandb.finish()

    # Save final results to log
    results = {"best_epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"], 
               "val_mae_m": ckpt["val_mae_m"], "test_loss": test_loss, 
               "test_mae_m": test_mae}
    with open(ROOT / "logs" / "phase4_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to logs/phase4_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb)
