"""
evaluate_v1_1.py -- Post-Training Evaluation for V1.1 Residual BiLSTM
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v1_1 import RVRBiLSTM_V1_1
from src.models.train_v1_1 import prepare_v1_1_dataloaders, compute_mae_metres
from src.models.dataset import TARGET_ZONES

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating V1.1 on: {device}")

    # 1. Data
    _, _, test_loader, feature_cols, scaler_X, scaler_y = prepare_v1_1_dataloaders()

    # 2. Model
    best_ckpt = ROOT / "models" / "best_bilstm_v1_1.pt"
    if not best_ckpt.exists():
        print(f"Error: {best_ckpt} not found!")
        return

    ckpt = torch.load(best_ckpt, map_location=device)
    model = RVRBiLSTM_V1_1(
        input_size=len(feature_cols),
        hidden_size=ckpt['config']['hidden_size'],
        num_layers=ckpt['config']['num_layers'],
        output_size=ckpt['config']['output_size'],
        dropout=ckpt['config']['dropout'],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3. Predict
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)

    # 4. Metrics
    mae_overall, preds_m, targets_m = compute_mae_metres(preds_scaled, targets_scaled, scaler_y)

    print("\n" + "="*50)
    print(f"{'Runway Zone':<15} | {'MAE (m)':<10}")
    print("-" * 50)
    
    for i, zone in enumerate(TARGET_ZONES):
        mae = mean_absolute_error(targets_m[:, i], preds_m[:, i])
        print(f"{zone:<15} | {mae:<10.2f}")
    
    print("-" * 50)
    print(f"{'OVERALL MAE':<15} | {mae_overall:<10.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
