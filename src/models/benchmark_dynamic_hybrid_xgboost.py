import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
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
XGB_MODEL_PATH = ROOT / "models" / "best_xgboost_multi_recall.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RVRWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 36):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        return self.features[idx : idx + self.seq_len], self.targets[idx + self.seq_len - 1]


def load_lstm_model(path: Path, input_size: int) -> RVRAttentionLSTM_V3:
    model = RVRAttentionLSTM_V3(input_size=input_size).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    return model


def collect_lstm_predictions(model: RVRAttentionLSTM_V3, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def calculate_metrics(pred_m: np.ndarray, true_m: np.ndarray, fog_threshold: float) -> dict[str, float]:
    mae = mean_absolute_error(true_m, pred_m)
    rmse = np.sqrt(mean_squared_error(true_m, pred_m))
    r2 = r2_score(true_m, pred_m)

    acc100 = np.mean(np.abs(true_m - pred_m) <= 100) * 100.0
    acc200 = np.mean(np.abs(true_m - pred_m) <= 200) * 100.0

    y_true = (true_m < fog_threshold).astype(int).ravel()
    y_pred = (pred_m < fog_threshold).astype(int).ravel()

    fog_precision = precision_score(y_true, y_pred, zero_division=0) * 100.0
    fog_recall = recall_score(y_true, y_pred, zero_division=0) * 100.0
    fog_f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Acc@100m": float(acc100),
        "Acc@200m": float(acc200),
        "Fog Precision": float(fog_precision),
        "Fog Recall": float(fog_recall),
        "Fog F1": float(fog_f1),
    }


def dynamic_blend_v3_v5(
    v3_scaled: np.ndarray,
    v5_scaled: np.ndarray,
    scaler_y,
    w_v5_clear: float,
    w_v5_fog: float,
    fog_lo: int,
    fog_hi: int,
) -> np.ndarray:
    v3_m = np.clip(scaler_y.inverse_transform(v3_scaled), 0, 10000)
    v5_m = np.clip(scaler_y.inverse_transform(v5_scaled), 0, 10000)
    min_pred_m = np.minimum(v3_m, v5_m)

    risk = np.clip((fog_hi - min_pred_m) / max(1.0, float(fog_hi - fog_lo)), 0.0, 1.0)
    w_v5 = w_v5_clear + (w_v5_fog - w_v5_clear) * risk
    w_v3 = 1.0 - w_v5
    return w_v3 * v3_scaled + w_v5 * v5_scaled


def tune_dynamic_v3_v5(v3_val_s: np.ndarray, v5_val_s: np.ndarray, y_val_s: np.ndarray, scaler_y, fog_threshold: float) -> dict:
    y_val_m = np.clip(scaler_y.inverse_transform(y_val_s), 0, 10000)
    v3_val_m = np.clip(scaler_y.inverse_transform(v3_val_s), 0, 10000)
    baseline_recall = calculate_metrics(v3_val_m, y_val_m, fog_threshold)["Fog Recall"]

    best = {
        "params": {
            "w_v5_clear": 0.25,
            "w_v5_fog": 0.60,
            "fog_lo": 600,
            "fog_hi": 1300,
        },
        "mae": float("inf"),
    }

    for w_v5_clear in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for w_v5_fog in [0.50, 0.60, 0.70, 0.80]:
            if w_v5_fog < w_v5_clear:
                continue
            for fog_lo in [400, 500, 600]:
                for fog_hi in [900, 1100, 1300]:
                    if fog_hi <= fog_lo:
                        continue

                    dyn_val_s = dynamic_blend_v3_v5(
                        v3_val_s,
                        v5_val_s,
                        scaler_y,
                        w_v5_clear=w_v5_clear,
                        w_v5_fog=w_v5_fog,
                        fog_lo=fog_lo,
                        fog_hi=fog_hi,
                    )
                    dyn_val_m = np.clip(scaler_y.inverse_transform(dyn_val_s), 0, 10000)
                    metrics = calculate_metrics(dyn_val_m, y_val_m, fog_threshold)

                    if metrics["Fog Recall"] >= baseline_recall and metrics["MAE"] < best["mae"]:
                        best = {
                            "params": {
                                "w_v5_clear": float(w_v5_clear),
                                "w_v5_fog": float(w_v5_fog),
                                "fog_lo": int(fog_lo),
                                "fog_hi": int(fog_hi),
                            },
                            "mae": float(metrics["MAE"]),
                        }

    return best


def blend_dynamic_with_xgb(
    dyn_m: np.ndarray,
    xgb_m: np.ndarray,
    w_xgb_clear: float,
    w_xgb_fog: float,
    fog_lo: int,
    fog_hi: int,
) -> np.ndarray:
    # Increase XGBoost influence in clearer regimes, reduce it during fog-like risk.
    min_pred_m = np.minimum(dyn_m, xgb_m)
    risk = np.clip((fog_hi - min_pred_m) / max(1.0, float(fog_hi - fog_lo)), 0.0, 1.0)
    w_xgb = w_xgb_clear + (w_xgb_fog - w_xgb_clear) * risk
    return (1.0 - w_xgb) * dyn_m + w_xgb * xgb_m


def tune_dynamic_xgb_fusion(dyn_val_m: np.ndarray, xgb_val_m: np.ndarray, y_val_m: np.ndarray, fog_threshold: float) -> dict:
    dyn_metrics = calculate_metrics(dyn_val_m, y_val_m, fog_threshold)

    best = {
        "params": {
            "w_xgb_clear": 0.30,
            "w_xgb_fog": 0.05,
            "fog_lo": 600,
            "fog_hi": 1300,
        },
        "score": float("-inf"),
        "metrics": dyn_metrics,
    }

    for w_xgb_clear in [0.10, 0.20, 0.30, 0.40, 0.50]:
        for w_xgb_fog in [0.00, 0.05, 0.10, 0.15, 0.20]:
            if w_xgb_fog > w_xgb_clear:
                continue
            for fog_lo in [400, 500, 600]:
                for fog_hi in [900, 1100, 1300]:
                    if fog_hi <= fog_lo:
                        continue

                    fused = blend_dynamic_with_xgb(
                        dyn_val_m,
                        xgb_val_m,
                        w_xgb_clear=w_xgb_clear,
                        w_xgb_fog=w_xgb_fog,
                        fog_lo=fog_lo,
                        fog_hi=fog_hi,
                    )
                    metrics = calculate_metrics(fused, y_val_m, fog_threshold)

                    # Prefer better F1 while discouraging large MAE regressions.
                    score = metrics["Fog F1"] - 0.0007 * max(0.0, metrics["MAE"] - dyn_metrics["MAE"])
                    if score > best["score"]:
                        best = {
                            "params": {
                                "w_xgb_clear": float(w_xgb_clear),
                                "w_xgb_fog": float(w_xgb_fog),
                                "fog_lo": int(fog_lo),
                                "fog_hi": int(fog_hi),
                            },
                            "score": float(score),
                            "metrics": metrics,
                        }

    return best


def fit_per_target_ridge_stack(
    y_val_m: np.ndarray,
    v3_val_m: np.ndarray,
    v5_val_m: np.ndarray,
    dyn_val_m: np.ndarray,
    xgb_val_m: np.ndarray,
    v3_test_m: np.ndarray,
    v5_test_m: np.ndarray,
    dyn_test_m: np.ndarray,
    xgb_test_m: np.ndarray,
) -> np.ndarray:
    n_targets = y_val_m.shape[1]
    stacked_test = np.zeros_like(xgb_test_m)

    for j in range(n_targets):
        x_meta_val = np.column_stack(
            [
                dyn_val_m[:, j],
                v3_val_m[:, j],
                v5_val_m[:, j],
                xgb_val_m[:, j],
            ]
        )
        x_meta_test = np.column_stack(
            [
                dyn_test_m[:, j],
                v3_test_m[:, j],
                v5_test_m[:, j],
                xgb_test_m[:, j],
            ]
        )

        reg = Ridge(alpha=1.0, random_state=42)
        reg.fit(x_meta_val, y_val_m[:, j])
        stacked_test[:, j] = reg.predict(x_meta_test)

    return np.clip(stacked_test, 0, 10000)


def print_table(results: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 133)
    print(
        "| Model                                | MAE (m)   | RMSE (m)  | R2      | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |"
    )
    print("| :---                                 | ---:      | ---:      | ---:    | ---:     | ---:     | ---:          | ---:       | ---:   |")

    for name, m in results.items():
        print(
            f"| {name:36} | "
            f"{m['MAE']:8.2f} | {m['RMSE']:8.2f} | {m['R2']:7.4f} | "
            f"{m['Acc@100m']:7.2f}% | {m['Acc@200m']:7.2f}% | "
            f"{m['Fog Precision']:12.2f}% | {m['Fog Recall']:9.2f}% | {m['Fog F1']:6.4f} |"
        )

    print("=" * 133)


def run_experiment(seq_len: int, fog_threshold: float, xgb_model_path: Path, batch_size: int) -> None:
    print("=" * 90)
    print("EXPERIMENT: Dynamic Hybrid + XGBoost Fusion")
    print("=" * 90)

    if not xgb_model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {xgb_model_path}. Train or provide --xgb-model-path first."
        )

    df = pd.read_parquet(DATA_PATH)
    df = df.select_dtypes(include=[np.number]).dropna()

    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    feature_cols = [c for c in df.columns if c not in target_cols]

    val_df = df[df.index.year == 2024].copy()
    test_df = df[df.index.year == 2025].copy()

    scaler_x = joblib.load(SCALER_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(SCALER_DIR / "scaler_y.pkl")

    val_x_scaled = scaler_x.transform(val_df[feature_cols])
    val_y_scaled = scaler_y.transform(val_df[target_cols])
    test_x_scaled = scaler_x.transform(test_df[feature_cols])
    test_y_scaled = scaler_y.transform(test_df[target_cols])

    val_loader = DataLoader(RVRWindowDataset(val_x_scaled, val_y_scaled, seq_len=seq_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(RVRWindowDataset(test_x_scaled, test_y_scaled, seq_len=seq_len), batch_size=batch_size, shuffle=False)

    input_size = len(feature_cols)
    model_v3 = load_lstm_model(MODEL_V3_PATH, input_size)
    model_v5 = load_lstm_model(MODEL_V5_PATH, input_size)

    v3_val_s, y_val_s = collect_lstm_predictions(model_v3, val_loader)
    v5_val_s, _ = collect_lstm_predictions(model_v5, val_loader)
    v3_test_s, y_test_s = collect_lstm_predictions(model_v3, test_loader)
    v5_test_s, _ = collect_lstm_predictions(model_v5, test_loader)

    best_dyn = tune_dynamic_v3_v5(v3_val_s, v5_val_s, y_val_s, scaler_y, fog_threshold)
    p_dyn = best_dyn["params"]

    dyn_val_s = dynamic_blend_v3_v5(
        v3_val_s,
        v5_val_s,
        scaler_y,
        w_v5_clear=p_dyn["w_v5_clear"],
        w_v5_fog=p_dyn["w_v5_fog"],
        fog_lo=p_dyn["fog_lo"],
        fog_hi=p_dyn["fog_hi"],
    )
    dyn_test_s = dynamic_blend_v3_v5(
        v3_test_s,
        v5_test_s,
        scaler_y,
        w_v5_clear=p_dyn["w_v5_clear"],
        w_v5_fog=p_dyn["w_v5_fog"],
        fog_lo=p_dyn["fog_lo"],
        fog_hi=p_dyn["fog_hi"],
    )

    y_val_m = np.clip(scaler_y.inverse_transform(y_val_s), 0, 10000)
    y_test_m = np.clip(scaler_y.inverse_transform(y_test_s), 0, 10000)

    v3_val_m = np.clip(scaler_y.inverse_transform(v3_val_s), 0, 10000)
    v5_val_m = np.clip(scaler_y.inverse_transform(v5_val_s), 0, 10000)
    v3_test_m = np.clip(scaler_y.inverse_transform(v3_test_s), 0, 10000)
    v5_test_m = np.clip(scaler_y.inverse_transform(v5_test_s), 0, 10000)
    dyn_val_m = np.clip(scaler_y.inverse_transform(dyn_val_s), 0, 10000)
    dyn_test_m = np.clip(scaler_y.inverse_transform(dyn_test_s), 0, 10000)

    xgb_bundle = joblib.load(xgb_model_path)
    estimators = xgb_bundle["estimators"]

    xgb_val_all = np.column_stack([est.predict(val_df[feature_cols].to_numpy(dtype=np.float32)) for est in estimators])
    xgb_test_all = np.column_stack([est.predict(test_df[feature_cols].to_numpy(dtype=np.float32)) for est in estimators])

    # Align with sequence targets used by LSTM windows.
    # RVRWindowDataset produces targets at indices [seq_len-1, ..., n-2].
    start_idx = seq_len - 1
    end_idx = -1
    xgb_val_m = np.clip(xgb_val_all[start_idx:end_idx], 0, 10000)
    xgb_test_m = np.clip(xgb_test_all[start_idx:end_idx], 0, 10000)

    if xgb_val_m.shape != y_val_m.shape:
        raise RuntimeError(f"Validation alignment mismatch: xgb={xgb_val_m.shape}, y={y_val_m.shape}")
    if xgb_test_m.shape != y_test_m.shape:
        raise RuntimeError(f"Test alignment mismatch: xgb={xgb_test_m.shape}, y={y_test_m.shape}")

    best_dyn_xgb = tune_dynamic_xgb_fusion(dyn_val_m, xgb_val_m, y_val_m, fog_threshold)
    p_dx = best_dyn_xgb["params"]

    dyn_xgb_test_m = blend_dynamic_with_xgb(
        dyn_test_m,
        xgb_test_m,
        w_xgb_clear=p_dx["w_xgb_clear"],
        w_xgb_fog=p_dx["w_xgb_fog"],
        fog_lo=p_dx["fog_lo"],
        fog_hi=p_dx["fog_hi"],
    )

    ridge_stack_test_m = fit_per_target_ridge_stack(
        y_val_m,
        v3_val_m,
        v5_val_m,
        dyn_val_m,
        xgb_val_m,
        v3_test_m,
        v5_test_m,
        dyn_test_m,
        xgb_test_m,
    )

    print(
        "Validation-selected dynamic V3/V5 params -> "
        f"w_v5_clear={p_dyn['w_v5_clear']:.2f}, w_v5_fog={p_dyn['w_v5_fog']:.2f}, "
        f"fog_lo={p_dyn['fog_lo']}, fog_hi={p_dyn['fog_hi']}"
    )
    print(
        "Validation-selected dynamic(Hybrid+XGB) params -> "
        f"w_xgb_clear={p_dx['w_xgb_clear']:.2f}, w_xgb_fog={p_dx['w_xgb_fog']:.2f}, "
        f"fog_lo={p_dx['fog_lo']}, fog_hi={p_dx['fog_hi']}"
    )

    results = {
        "V3.1": calculate_metrics(v3_test_m, y_test_m, fog_threshold),
        "V5": calculate_metrics(v5_test_m, y_test_m, fog_threshold),
        "Dynamic Hybrid (V3+V5)": calculate_metrics(dyn_test_m, y_test_m, fog_threshold),
        "XGBoost": calculate_metrics(xgb_test_m, y_test_m, fog_threshold),
        "Dynamic Hybrid + XGBoost": calculate_metrics(dyn_xgb_test_m, y_test_m, fog_threshold),
        "Ridge Stacking (Dyn,V3,V5,XGB)": calculate_metrics(ridge_stack_test_m, y_test_m, fog_threshold),
    }

    print_table(results)

    base = results["Dynamic Hybrid (V3+V5)"]
    fused = results["Dynamic Hybrid + XGBoost"]
    stack = results["Ridge Stacking (Dyn,V3,V5,XGB)"]

    print("\nDelta vs Dynamic Hybrid baseline:")
    print(f"- Dynamic+XGB: MAE {fused['MAE'] - base['MAE']:+.2f} m | Precision {fused['Fog Precision'] - base['Fog Precision']:+.2f}% | Recall {fused['Fog Recall'] - base['Fog Recall']:+.2f}% | F1 {fused['Fog F1'] - base['Fog F1']:+.4f}")
    print(f"- Ridge Stack: MAE {stack['MAE'] - base['MAE']:+.2f} m | Precision {stack['Fog Precision'] - base['Fog Precision']:+.2f}% | Recall {stack['Fog Recall'] - base['Fog Recall']:+.2f}% | F1 {stack['Fog F1'] - base['Fog F1']:+.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fusion of Dynamic Hybrid (V3+V5) with XGBoost.")
    parser.add_argument("--seq-len", type=int, default=36, help="Sequence length used for LSTM windows")
    parser.add_argument("--fog-threshold", type=float, default=600.0, help="Fog threshold in meters")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for LSTM inference")
    parser.add_argument(
        "--xgb-model-path",
        type=Path,
        default=XGB_MODEL_PATH,
        help="Path to xgboost model bundle (*.joblib)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        seq_len=args.seq_len,
        fog_threshold=args.fog_threshold,
        xgb_model_path=args.xgb_model_path,
        batch_size=args.batch_size,
    )