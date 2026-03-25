"""
evaluate_results.py — Phase 5 Granular Evaluation for IGIA RVR

Generates:
  - Zone-wise MAE comparisons
  - Prediction vs. Actual time-series plots for 2025 Test Set
  - Scatter plots of prediction accuracy
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model import RVRBiLSTM
from src.models.dataset import prepare_dataloaders, TARGET_ZONES, TARGET_COLS

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load Data
    _, _, test_loader, feature_cols, scaler_X, scaler_y = prepare_dataloaders(
        parquet_path=str(ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet"),
        scaler_dir=str(ROOT / "data" / "processed" / "scalers"),
        batch_size=512,
        num_workers=0  # Faster for small eval
    )
    
    # 2. Load Model
    checkpoint = torch.load(ROOT / "models" / "best_bilstm.pt", map_location=device)
    model = RVRBiLSTM(
        input_size=len(feature_cols),
        hidden_size=checkpoint['config']['hidden_size'],
        num_layers=checkpoint['config']['num_layers'],
        output_size=checkpoint['config']['output_size']
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 3. Predict
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # 4. Inverse Scaling to Real Metres
    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)
    
    preds_m = scaler_y.inverse_transform(preds_scaled)
    targets_m = scaler_y.inverse_transform(targets_scaled)
    
    # 5. Calculate Zone-wise MAE
    zone_results = {}
    print("\n" + "="*40)
    print(f"{'Runway Zone':<15} | {'MAE (m)':<10} | {'R2 Score':<10}")
    print("-" * 40)
    
    for i, zone in enumerate(TARGET_ZONES):
        mae = mean_absolute_error(targets_m[:, i], preds_m[:, i])
        r2 = r2_score(targets_m[:, i], preds_m[:, i])
        zone_results[zone] = {"MAE": round(mae, 2), "R2": round(r2, 4)}
        print(f"{zone:<15} | {mae:<10.2f} | {r2:<10.4f}")
    
    avg_mae = mean_absolute_error(targets_m, preds_m)
    print("-" * 40)
    print(f"{'AVERAGE':<15} | {avg_mae:<10.2f}")
    print("="*40)

    # 6. Visualization: 24-hour Fog Event comparison (First 144 steps = ~1 day)
    # Let's pick a window where RVR is low (<1000m) to see fog capture
    # Searching for a fog event in the test set:
    low_vis_idx = np.where(targets_m[:, 0] < 1000)[0]
    if len(low_vis_idx) > 0:
        start_idx = max(0, low_vis_idx[0] - 50)
        end_idx = min(len(targets_m), start_idx + 200)
    else:
        start_idx, end_idx = 0, 200

    plt.figure(figsize=(15, 7))
    # Plot for Zone 09 TDZ
    plt.plot(targets_m[start_idx:end_idx, 0], label='Actual RVR (09 TDZ)', color='black', alpha=0.5, linestyle='--')
    plt.plot(preds_m[start_idx:end_idx, 0], label='Predicted RVR (6h Ahead)', color='blue', linewidth=2)
    
    plt.title("BiLSTM: 6-Hour RVR Forecast vs Actual (IGIA Zone 09 TDZ)")
    plt.xlabel("10-Minute Intervals")
    plt.ylabel("Runway Visual Range (Metres)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = ROOT / "logs" / "phase5_rvr_forecast_plot.png"
    plt.savefig(output_plot)
    print(f"\nEvaluation plot saved to {output_plot}")
    
    # 7. Correlation Analysis
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_m[:, 0], preds_m[:, 0], alpha=0.1, s=1)
    plt.plot([0, 5000], [0, 5000], 'r--')
    plt.title("Prediction vs Actual Correlation (All Test Samples)")
    plt.xlabel("Actual RVR (m)")
    plt.ylabel("Predicted RVR (m)")
    plt.savefig(ROOT / "logs" / "phase5_correlation_scatter.png")

if __name__ == "__main__":
    run_evaluation()
