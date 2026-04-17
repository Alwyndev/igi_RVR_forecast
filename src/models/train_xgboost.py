"""
train_xgboost.py -- Multi-horizon XGBoost baseline for IGIA RVR forecasting.

Trains a separate boosted-tree regressor per target using sklearn's MultiOutputRegressor,
with the same data split convention used by the LSTM pipeline:
- Train: <= 2023
- Val:   2024
- Test:  2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from tqdm import tqdm

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install dependencies with: pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parents[2]


CONFIG = {
    "parquet_path": ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet",
    "model_out": ROOT / "models" / "best_xgboost_multi.joblib",
    "meta_out": ROOT / "models" / "best_xgboost_multi.meta.json",
    "fog_threshold_m": 600,
    "fog_weight": 4.0,
    "multioutput_jobs": 6,
    "verbose_eval_step": 100,
    "xgb": {
        "n_estimators": 900,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "min_child_weight": 4,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
    },
}


def _load_dataframe(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex for year-based splits.")

    return df


def _select_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    target_cols = sorted([c for c in numeric_df.columns if c.startswith("target_")])
    if len(target_cols) != 50:
        raise ValueError(f"Expected 50 target columns, found {len(target_cols)}")

    feature_cols = [c for c in numeric_df.columns if c not in target_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding targets.")

    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.dropna(inplace=True)

    return numeric_df, feature_cols, target_cols


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df.index.year <= 2023]
    val_df = df[df.index.year == 2024]
    test_df = df[df.index.year == 2025]

    if len(train_df) == 0:
        raise ValueError("Train split is empty. Check datetime index and data coverage.")
    if len(val_df) == 0:
        raise ValueError("Validation split is empty. Check datetime index and data coverage.")
    if len(test_df) == 0:
        raise ValueError("Test split is empty. Check datetime index and data coverage.")

    return train_df, val_df, test_df


def _make_sample_weight(y: np.ndarray, fog_threshold_m: float, fog_weight: float) -> np.ndarray:
    # Increase weight when any horizon/zone indicates fog to improve fog-event sensitivity.
    fog_rows = (y < fog_threshold_m).any(axis=1)
    weights = np.ones(len(y), dtype=np.float32)
    weights[fog_rows] = fog_weight
    return weights


def _evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, fog_threshold_m: float) -> dict[str, float]:
    y_pred_clip = np.clip(y_pred, 0, 10000)
    y_true_clip = np.clip(y_true, 0, 10000)

    mae_m = mean_absolute_error(y_true_clip, y_pred_clip)

    y_true_fog = (y_true_clip < fog_threshold_m).astype(int).ravel()
    y_pred_fog = (y_pred_clip < fog_threshold_m).astype(int).ravel()

    precision = precision_score(y_true_fog, y_pred_fog, zero_division=0)
    recall = recall_score(y_true_fog, y_pred_fog, zero_division=0)

    return {
        "mae_m": float(mae_m),
        "precision_at_600m": float(precision),
        "recall_at_600m": float(recall),
    }


def train_xgboost(parquet_path: Path, model_out: Path, meta_out: Path) -> None:
    print(f"Loading dataset: {parquet_path}")
    df = _load_dataframe(parquet_path)
    df, feature_cols, target_cols = _select_features_targets(df)
    train_df, val_df, test_df = _time_split(df)

    print(
        "Split sizes -> "
        f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )
    print(f"Features: {len(feature_cols)} | Targets: {len(target_cols)}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_cols].to_numpy(dtype=np.float32)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_cols].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_cols].to_numpy(dtype=np.float32)

    sample_weight = _make_sample_weight(
        y_train,
        fog_threshold_m=CONFIG["fog_threshold_m"],
        fog_weight=CONFIG["fog_weight"],
    )

    print("Training XGBoost models target-by-target with live progress...")
    print(
        f"Boosting rounds per target: {CONFIG['xgb']['n_estimators']} | "
        f"Targets: {len(target_cols)}"
    )

    estimators: list[XGBRegressor] = []
    for idx, target_name in enumerate(tqdm(target_cols, desc="Targets", unit="target"), start=1):
        print(f"\n[{idx:02d}/{len(target_cols)}] Fitting {target_name}")
        est = XGBRegressor(**CONFIG["xgb"])
        est.fit(
            X_train,
            y_train[:, idx - 1],
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val[:, idx - 1])],
            verbose=CONFIG["verbose_eval_step"],
        )
        estimators.append(est)

    print("Evaluating on validation and test splits...")
    val_pred = np.column_stack([est.predict(X_val) for est in estimators])
    test_pred = np.column_stack([est.predict(X_test) for est in estimators])

    val_metrics = _evaluate_metrics(y_val, val_pred, fog_threshold_m=CONFIG["fog_threshold_m"])
    test_metrics = _evaluate_metrics(y_test, test_pred, fog_threshold_m=CONFIG["fog_threshold_m"])

    print(
        "Validation -> "
        f"MAE: {val_metrics['mae_m']:.2f} m | "
        f"Precision@600m: {val_metrics['precision_at_600m']:.2%} | "
        f"Recall@600m: {val_metrics['recall_at_600m']:.2%}"
    )
    print(
        "Test -> "
        f"MAE: {test_metrics['mae_m']:.2f} m | "
        f"Precision@600m: {test_metrics['precision_at_600m']:.2%} | "
        f"Recall@600m: {test_metrics['recall_at_600m']:.2%}"
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model_type": "xgboost_multi_target_separate",
        "estimators": estimators,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }
    joblib.dump(model_bundle, model_out)

    metadata = {
        "model_type": "MultiOutputRegressor(XGBRegressor)",
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "fog_threshold_m": CONFIG["fog_threshold_m"],
        "fog_weight": CONFIG["fog_weight"],
        "xgb_params": CONFIG["xgb"],
        "multioutput_jobs": CONFIG["multioutput_jobs"],
        "verbose_eval_step": CONFIG["verbose_eval_step"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved metadata: {meta_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model for IGIA multi-horizon RVR forecasting.")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=CONFIG["parquet_path"],
        help="Path to igia_rvr_training_dataset_multi.parquet",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=CONFIG["model_out"],
        help="Where to save trained joblib model",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=CONFIG["meta_out"],
        help="Where to save training metadata and metrics JSON",
    )
    parser.add_argument(
        "--fog-weight",
        type=float,
        default=CONFIG["fog_weight"],
        help="Sample-weight multiplier for rows containing fog (<600m in any target)",
    )
    parser.add_argument(
        "--multioutput-jobs",
        type=int,
        default=CONFIG["multioutput_jobs"],
        help="Parallel jobs across output targets",
    )
    parser.add_argument(
        "--verbose-eval-step",
        type=int,
        default=CONFIG["verbose_eval_step"],
        help="Print eval metric every N boosting rounds for each target",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=CONFIG["xgb"]["n_estimators"],
        help="Boosting rounds per target model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG["fog_weight"] = float(args.fog_weight)
    CONFIG["multioutput_jobs"] = int(args.multioutput_jobs)
    CONFIG["xgb"]["n_estimators"] = int(args.n_estimators)
    CONFIG["verbose_eval_step"] = int(args.verbose_eval_step)

    train_xgboost(
        parquet_path=Path(args.parquet_path),
        model_out=Path(args.model_out),
        meta_out=Path(args.meta_out),
    )


if __name__ == "__main__":
    main()
