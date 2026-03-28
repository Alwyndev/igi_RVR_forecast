import os
import sys
from pathlib import Path

import joblib
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

# -- Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3

# -- Config
DATA_PATH = ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"
MODEL_V3_PATH = ROOT / "models" / "best_lstm_v3.pt"
MODEL_V5_PATH = ROOT / "models" / "best_lstm_v5.pt"
SCALER_DIR = ROOT / "data" / "processed" / "scalers_v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOG_THRESHOLD = 600
SEQ_LEN = 36


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


def calculate_metrics(pred_m, true_m):
    mae = mean_absolute_error(true_m, pred_m)
    rmse = np.sqrt(mean_squared_error(true_m, pred_m))
    r2 = r2_score(true_m, pred_m)

    acc100 = np.mean(np.abs(true_m - pred_m) <= 100) * 100
    acc200 = np.mean(np.abs(true_m - pred_m) <= 200) * 100

    y_true = (true_m < FOG_THRESHOLD).astype(int).flatten()
    y_pred = (pred_m < FOG_THRESHOLD).astype(int).flatten()
    fog_precision = precision_score(y_true, y_pred, zero_division=0) * 100
    fog_recall = recall_score(y_true, y_pred, zero_division=0) * 100
    fog_f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Acc@100m": acc100,
        "Acc@200m": acc200,
        "Fog Precision": fog_precision,
        "Fog Recall": fog_recall,
        "Fog F1": fog_f1,
    }


def blend_predictions(v3_scaled, v5_scaled, w_v3):
    w_v5 = 1.0 - w_v3
    return w_v3 * v3_scaled + w_v5 * v5_scaled


def dynamic_blend_predictions(v3_scaled, v5_scaled, scaler_y, w_v5_clear, w_v5_fog, fog_lo, fog_hi):
    # Build a smooth fog-risk weight from model-predicted RVR in meters.
    v3_m = np.clip(scaler_y.inverse_transform(v3_scaled), 0, 10000)
    v5_m = np.clip(scaler_y.inverse_transform(v5_scaled), 0, 10000)
    min_pred_m = np.minimum(v3_m, v5_m)

    # risk in [0,1]: 0=clear, 1=fog-like
    risk = np.clip((fog_hi - min_pred_m) / max(1.0, (fog_hi - fog_lo)), 0.0, 1.0)
    w_v5 = w_v5_clear + (w_v5_fog - w_v5_clear) * risk
    w_v3 = 1.0 - w_v5

    return w_v3 * v3_scaled + w_v5 * v5_scaled


def find_best_weight_for_mae(v3_val, v5_val, y_val):
    weights = np.arange(0.0, 1.01, 0.05)
    best = {"w": 0.5, "mae": float("inf")}
    for w in weights:
        pred = blend_predictions(v3_val, v5_val, w)
        mae = mean_absolute_error(y_val, pred)
        if mae < best["mae"]:
            best = {"w": float(w), "mae": float(mae)}
    return best


def find_best_dynamic_params(v3_val, v5_val, y_val, scaler_y):
    y_val_m = np.clip(scaler_y.inverse_transform(y_val), 0, 10000)
    baseline_recall = calculate_metrics(
        np.clip(scaler_y.inverse_transform(v3_val), 0, 10000),
        y_val_m,
    )["Fog Recall"]

    best = {
        "params": {
            "w_v5_clear": 0.20,
            "w_v5_fog": 0.60,
            "fog_lo": 500,
            "fog_hi": 1100,
        },
        "mae": float("inf"),
        "recall": 0.0,
    }

    for w_v5_clear in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for w_v5_fog in [0.50, 0.60, 0.70, 0.80]:
            if w_v5_fog < w_v5_clear:
                continue
            for fog_lo in [400, 500, 600]:
                for fog_hi in [900, 1100, 1300]:
                    if fog_hi <= fog_lo:
                        continue

                    dyn_pred_s = dynamic_blend_predictions(
                        v3_val,
                        v5_val,
                        scaler_y,
                        w_v5_clear=w_v5_clear,
                        w_v5_fog=w_v5_fog,
                        fog_lo=fog_lo,
                        fog_hi=fog_hi,
                    )
                    dyn_pred_m = np.clip(scaler_y.inverse_transform(dyn_pred_s), 0, 10000)
                    metrics = calculate_metrics(dyn_pred_m, y_val_m)

                    # Keep safety no worse than V3 on validation, then minimize MAE.
                    if metrics["Fog Recall"] >= baseline_recall and metrics["MAE"] < best["mae"]:
                        best = {
                            "params": {
                                "w_v5_clear": float(w_v5_clear),
                                "w_v5_fog": float(w_v5_fog),
                                "fog_lo": int(fog_lo),
                                "fog_hi": int(fog_hi),
                            },
                            "mae": float(metrics["MAE"]),
                            "recall": float(metrics["Fog Recall"]),
                        }

    return best


def print_table(results):
    print("\n" + "=" * 105)
    print("| Metric                    | V3.1              | V5                | Static Hybrid     | Dynamic Hybrid    | Winner           |")
    print("| :---                      | :---:             | :---:             | :---:             | :---:             | :---:            |")

    metrics = [
        ("Absolute Error (MAE)", "MAE", False),
        ("Root Mean Sq Err (RMSE)", "RMSE", False),
        ("Coefficient of Det (R2)", "R2", True),
        ("Accuracy within 100m", "Acc@100m", True),
        ("Accuracy within 200m", "Acc@200m", True),
        ("Fog Precision (<600m)", "Fog Precision", True),
        ("Fog Recall (Sensitivity)", "Fog Recall", True),
        ("Fog F1-Score (Balanced)", "Fog F1", True),
    ]

    for label, key, higher_better in metrics:
        v3 = results["V3.1"][key]
        v5 = results["V5"][key]
        hy_static = results["Static Hybrid"][key]
        hy_dynamic = results["Dynamic Hybrid"][key]

        pack = {"V3.1": v3, "V5": v5, "Static Hybrid": hy_static, "Dynamic Hybrid": hy_dynamic}
        winner = max(pack, key=pack.get) if higher_better else min(pack, key=pack.get)

        if key in ["R2", "Fog F1"]:
            fmt_v3 = f"{v3:8.4f}"
            fmt_v5 = f"{v5:8.4f}"
            fmt_hy_static = f"{hy_static:8.4f}"
            fmt_hy_dynamic = f"{hy_dynamic:8.4f}"
        elif key in ["Acc@100m", "Acc@200m", "Fog Precision", "Fog Recall"]:
            fmt_v3 = f"{v3:7.2f}%"
            fmt_v5 = f"{v5:7.2f}%"
            fmt_hy_static = f"{hy_static:7.2f}%"
            fmt_hy_dynamic = f"{hy_dynamic:7.2f}%"
        else:
            fmt_v3 = f"{v3:8.2f}"
            fmt_v5 = f"{v5:8.2f}"
            fmt_hy_static = f"{hy_static:8.2f}"
            fmt_hy_dynamic = f"{hy_dynamic:8.2f}"

        print(
            f"| {label:25} | {fmt_v3:17} | {fmt_v5:17} | {fmt_hy_static:17} | {fmt_hy_dynamic:17} | {winner:16} |"
        )

    print("=" * 105)


def run_comparison():
    print("=" * 72)
    print("HYBRID BENCHMARK: V3.1, V5, Static Blend, and Dynamic Blend")
    print("=" * 72)

    df = pd.read_parquet(DATA_PATH)
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols]

    val_df = df[df.index.year == 2024].copy()
    test_df = df[df.index.year == 2025].copy()

    scaler_X = joblib.load(SCALER_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(SCALER_DIR / "scaler_y.pkl")

    vX = scaler_X.transform(val_df[feature_cols])
    vy = scaler_y.transform(val_df[target_cols])
    tX = scaler_X.transform(test_df[feature_cols])
    ty = scaler_y.transform(test_df[target_cols])

    val_loader = DataLoader(RVRWindowDataset(vX, vy, seq_len=SEQ_LEN), batch_size=256, shuffle=False)
    test_loader = DataLoader(RVRWindowDataset(tX, ty, seq_len=SEQ_LEN), batch_size=256, shuffle=False)

    input_size = len(feature_cols)
    model_v3 = load_model(MODEL_V3_PATH, input_size)
    model_v5 = load_model(MODEL_V5_PATH, input_size)

    # Get validation predictions in scaled space for blend-weight selection.
    v3_val_pred_s, val_true_s = collect_predictions(model_v3, val_loader)
    v5_val_pred_s, _ = collect_predictions(model_v5, val_loader)

    best_static = find_best_weight_for_mae(v3_val_pred_s, v5_val_pred_s, val_true_s)
    w_v3 = best_static["w"]
    w_v5 = 1.0 - w_v3

    best_dynamic = find_best_dynamic_params(v3_val_pred_s, v5_val_pred_s, val_true_s, scaler_y)
    p = best_dynamic["params"]
    print(f"Validation-selected static blend (MAE) -> w_v3={w_v3:.2f}, w_v5={w_v5:.2f}")
    print(
        "Validation-selected dynamic blend -> "
        f"w_v5_clear={p['w_v5_clear']:.2f}, w_v5_fog={p['w_v5_fog']:.2f}, "
        f"fog_lo={p['fog_lo']}m, fog_hi={p['fog_hi']}m"
    )

    # Evaluate on test split.
    v3_test_pred_s, test_true_s = collect_predictions(model_v3, test_loader)
    v5_test_pred_s, _ = collect_predictions(model_v5, test_loader)
    hy_static_test_pred_s = blend_predictions(v3_test_pred_s, v5_test_pred_s, w_v3)
    hy_dynamic_test_pred_s = dynamic_blend_predictions(
        v3_test_pred_s,
        v5_test_pred_s,
        scaler_y,
        w_v5_clear=p["w_v5_clear"],
        w_v5_fog=p["w_v5_fog"],
        fog_lo=p["fog_lo"],
        fog_hi=p["fog_hi"],
    )

    v3_test_pred_m = np.clip(scaler_y.inverse_transform(v3_test_pred_s), 0, 10000)
    v5_test_pred_m = np.clip(scaler_y.inverse_transform(v5_test_pred_s), 0, 10000)
    hy_static_test_pred_m = np.clip(scaler_y.inverse_transform(hy_static_test_pred_s), 0, 10000)
    hy_dynamic_test_pred_m = np.clip(scaler_y.inverse_transform(hy_dynamic_test_pred_s), 0, 10000)
    test_true_m = np.clip(scaler_y.inverse_transform(test_true_s), 0, 10000)

    results = {
        "V3.1": calculate_metrics(v3_test_pred_m, test_true_m),
        "V5": calculate_metrics(v5_test_pred_m, test_true_m),
        "Static Hybrid": calculate_metrics(hy_static_test_pred_m, test_true_m),
        "Dynamic Hybrid": calculate_metrics(hy_dynamic_test_pred_m, test_true_m),
    }

    print_table(results)

    print("\nStatic hybrid summary:")
    print(f"- MAE delta vs V3.1: {results['Static Hybrid']['MAE'] - results['V3.1']['MAE']:+.2f} m")
    print(f"- MAE delta vs V5  : {results['Static Hybrid']['MAE'] - results['V5']['MAE']:+.2f} m")
    print(
        f"- Fog Recall delta vs V3.1: "
        f"{results['Static Hybrid']['Fog Recall'] - results['V3.1']['Fog Recall']:+.2f}%"
    )
    print(
        f"- Fog Recall delta vs V5  : "
        f"{results['Static Hybrid']['Fog Recall'] - results['V5']['Fog Recall']:+.2f}%"
    )

    print("\nDynamic hybrid summary:")
    print(f"- MAE delta vs V3.1: {results['Dynamic Hybrid']['MAE'] - results['V3.1']['MAE']:+.2f} m")
    print(f"- MAE delta vs V5  : {results['Dynamic Hybrid']['MAE'] - results['V5']['MAE']:+.2f} m")
    print(
        f"- Fog Recall delta vs V3.1: "
        f"{results['Dynamic Hybrid']['Fog Recall'] - results['V3.1']['Fog Recall']:+.2f}%"
    )
    print(
        f"- Fog Recall delta vs V5  : "
        f"{results['Dynamic Hybrid']['Fog Recall'] - results['V5']['Fog Recall']:+.2f}%"
    )


if __name__ == "__main__":
    run_comparison()
