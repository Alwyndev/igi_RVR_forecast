import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

DATA_PATH = ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"
SCALER_DIR = ROOT / "data" / "processed" / "scalers_v3"
MODEL_V3_PATH = ROOT / "models" / "best_lstm_v3.pt"
MODEL_V5_PATH = ROOT / "models" / "best_lstm_v5.pt"
LOG_DIR = ROOT / "logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOG_THRESHOLD = 600
SEQ_LEN = 36

# Dynamic-hybrid parameters selected from validation search.
W_V5_CLEAR = 0.25
W_V5_FOG = 0.60
FOG_LO = 600.0
FOG_HI = 1300.0
SELECTED_ZONES = ["09_TDZ", "11_TDZ", "MID_2810", "MID_2911"]
TIMESERIES_HORIZON = "1h"


class RVRWindowDataset(Dataset):
    def __init__(self, features, targets, seq_len=36):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        return self.features[idx : idx + self.seq_len], self.targets[idx + self.seq_len - 1]


def load_model(path, input_size):
    model = RVRAttentionLSTM_V3(input_size=input_size).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    return model


def collect_predictions(model, loader):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def dynamic_blend_m(v3_m, v5_m):
    min_pred = np.minimum(v3_m, v5_m)
    risk = np.clip((FOG_HI - min_pred) / (FOG_HI - FOG_LO), 0.0, 1.0)
    w_v5 = W_V5_CLEAR + (W_V5_FOG - W_V5_CLEAR) * risk
    w_v3 = 1.0 - w_v5
    return (w_v3 * v3_m) + (w_v5 * v5_m)


def calculate_metrics(pred_m, true_m):
    y_true = (true_m < FOG_THRESHOLD).astype(int).flatten()
    y_pred = (pred_m < FOG_THRESHOLD).astype(int).flatten()

    return {
        "MAE": mean_absolute_error(true_m, pred_m),
        "RMSE": np.sqrt(mean_squared_error(true_m, pred_m)),
        "R2": r2_score(true_m, pred_m),
        "Acc@100m": np.mean(np.abs(true_m - pred_m) <= 100) * 100,
        "Acc@200m": np.mean(np.abs(true_m - pred_m) <= 200) * 100,
        "Fog Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Fog Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "Fog F1": f1_score(y_true, y_pred, zero_division=0),
    }


def get_selected_indices(target_names, zones):
    selected = []
    for i, name in enumerate(target_names):
        if any(name.startswith(f"target_{z}_") for z in zones):
            selected.append(i)
    return selected


def plot_metrics_bar(metrics_by_model):
    metric_keys = ["MAE", "RMSE", "R2", "Acc@100m", "Acc@200m", "Fog Precision", "Fog Recall", "Fog F1"]
    model_names = ["V3.1", "V5", "Hybrid Dynamic"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, key in enumerate(metric_keys):
        vals = [metrics_by_model[m][key] for m in model_names]
        axes[i].bar(model_names, vals, color=["#1f77b4", "#d62728", "#2ca02c"])
        axes[i].set_title(key)
        axes[i].tick_params(axis="x", rotation=20)
        axes[i].grid(alpha=0.2)

    fig.suptitle("V3.1 vs V5 vs Dynamic Hybrid - Metric Comparison (2 TDZ + 2 MID)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = LOG_DIR / "benchmark_v3_v5_dynamic_metrics_grid_2tdz_2mid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_timeseries_4zones(actual_m, v3_m, v5_m, hy_m, target_names, zones, horizon):
    n = min(500, len(actual_m))
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, zone in zip(axes, zones):
        target_key = f"target_{zone}_rvr_actual_mean_{horizon}"
        idx = target_names.index(target_key)

        ax.plot(x, actual_m[:n, idx], color="gray", linestyle="--", label="Actual", alpha=0.9)
        ax.plot(x, v3_m[:n, idx], color="#1f77b4", label="V3.1", alpha=0.9)
        ax.plot(x, v5_m[:n, idx], color="#d62728", label="V5", alpha=0.9)
        ax.plot(x, hy_m[:n, idx], color="#2ca02c", linewidth=2.0, label="Dynamic Hybrid", alpha=0.95)
        ax.set_title(f"{zone} ({horizon})")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("RVR (meters)")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.suptitle("Actual vs V3.1 vs V5 vs Dynamic Hybrid (2 TDZ + 2 MID)", fontsize=15)
    out_path = LOG_DIR / "benchmark_v3_v5_dynamic_timeseries_2tdz_2mid_1h.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _density_curve(errors, bins):
    hist, edges = np.histogram(errors, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def plot_error_distribution(err_v3, err_v5, err_hy):
    low = np.percentile(np.concatenate([err_v3, err_v5, err_hy]), 1)
    high = np.percentile(np.concatenate([err_v3, err_v5, err_hy]), 99)
    bins = np.linspace(low, high, 120)

    x3, y3 = _density_curve(err_v3, bins)
    x5, y5 = _density_curve(err_v5, bins)
    xh, yh = _density_curve(err_hy, bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x3, y3, color="#1f77b4", label="V3.1 Error")
    ax.plot(x5, y5, color="#d62728", label="V5 Error")
    ax.plot(xh, yh, color="#2ca02c", label="Dynamic Hybrid Error")
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)

    ax.set_title("Error Distribution (Prediction - Actual) - 2 TDZ + 2 MID")
    ax.set_xlabel("Error (meters)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()

    out_path = LOG_DIR / "benchmark_v3_v5_dynamic_error_distribution_2tdz_2mid.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    df = df.select_dtypes(include=[np.number]).dropna()

    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols]

    test_df = df[df.index.year == 2025].copy()

    scaler_X = joblib.load(SCALER_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(SCALER_DIR / "scaler_y.pkl")

    tX = scaler_X.transform(test_df[feature_cols])
    ty = scaler_y.transform(test_df[target_cols])

    loader = DataLoader(RVRWindowDataset(tX, ty, seq_len=SEQ_LEN), batch_size=256, shuffle=False)

    model_v3 = load_model(MODEL_V3_PATH, len(feature_cols))
    model_v5 = load_model(MODEL_V5_PATH, len(feature_cols))

    v3_pred_s, true_s = collect_predictions(model_v3, loader)
    v5_pred_s, _ = collect_predictions(model_v5, loader)

    v3_m = np.clip(scaler_y.inverse_transform(v3_pred_s), 0, 10000)
    v5_m = np.clip(scaler_y.inverse_transform(v5_pred_s), 0, 10000)
    true_m = np.clip(scaler_y.inverse_transform(true_s), 0, 10000)
    hy_m = np.clip(dynamic_blend_m(v3_m, v5_m), 0, 10000)

    sel_idx = get_selected_indices(target_cols, SELECTED_ZONES)
    if not sel_idx:
        raise RuntimeError("No matching target columns found for selected zones.")

    true_sel = true_m[:, sel_idx]
    v3_sel = v3_m[:, sel_idx]
    v5_sel = v5_m[:, sel_idx]
    hy_sel = hy_m[:, sel_idx]

    metrics = {
        "V3.1": calculate_metrics(v3_sel, true_sel),
        "V5": calculate_metrics(v5_sel, true_sel),
        "Hybrid Dynamic": calculate_metrics(hy_sel, true_sel),
    }

    print("\nSubset benchmark metrics (2 TDZ + 2 MID):")
    header = (
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "Acc@100m",
        "Acc@200m",
        "Fog Precision",
        "Fog Recall",
        "Fog F1",
    )
    print("| " + " | ".join(header) + " |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name in ["V3.1", "V5", "Hybrid Dynamic"]:
        m = metrics[name]
        print(
            f"| {name} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} | "
            f"{m['Acc@100m']:.2f}% | {m['Acc@200m']:.2f}% | {m['Fog Precision']:.2f}% | "
            f"{m['Fog Recall']:.2f}% | {m['Fog F1']:.4f} |"
        )

    p1 = plot_metrics_bar(metrics)
    p2 = plot_timeseries_4zones(true_m, v3_m, v5_m, hy_m, target_cols, SELECTED_ZONES, TIMESERIES_HORIZON)
    p3 = plot_error_distribution((v3_sel - true_sel).flatten(), (v5_sel - true_sel).flatten(), (hy_sel - true_sel).flatten())

    print("Generated comparison graphs:")
    print(f"- Zones used: {', '.join(SELECTED_ZONES)}")
    print(f"- {p1}")
    print(f"- {p2}")
    print(f"- {p3}")


if __name__ == "__main__":
    main()
