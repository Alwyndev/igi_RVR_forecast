"""
Pseudo-realtime inference loop for airport simulation.

Reads synthetic stream row-by-row in 10-minute cadence, performs cleaning + feature
engineering to match the model's 104-feature expectation, runs V3 model inference,
updates a map, and writes MAE/RMSE/R2 metrics to separate log files.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.runway_config import CONSOLIDATED_ZONES
from src.models.model_v3 import RVRAttentionLSTM_V3

ROOT = Path(__file__).resolve().parents[2]

HORIZONS = ["10m", "30m", "1h", "3h", "6h"]
HORIZON_TO_STEPS = {"10m": 1, "30m": 3, "1h": 6, "3h": 18, "6h": 36}

STEPS_1_HOUR = 6
STEPS_2_HOURS = 12
STEPS_3_HOURS = 18
STEPS_6_HOURS = 36

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "sim_realtime_pipeline.log", encoding="utf-8"),
        ],
    )


def _load_coords(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_status_color(rvr_m: float) -> str:
    if rvr_m >= 1500:
        return "#006400"
    if rvr_m >= 550:
        return "#FFA500"
    if rvr_m >= 175:
        return "#FF0000"
    return "#000000"


class V3RealtimeEngine:
    def __init__(self, model_path: Path, scaler_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler_X = joblib.load(scaler_dir / "scaler_X.pkl")
        self.scaler_y = joblib.load(scaler_dir / "scaler_y.pkl")

        self.model = RVRAttentionLSTM_V3(
            input_size=104,
            hidden_size=384,
            num_layers=3,
            output_size=50,
            dropout=0.3,
        ).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_cols = list(self.scaler_X.feature_names_in_)
        self.target_names = sorted([
            f"target_{z}_rvr_actual_mean_{h}" for z in CONSOLIDATED_ZONES for h in HORIZONS
        ])

    def predict(self, engineered_df: pd.DataFrame) -> np.ndarray:
        if len(engineered_df) < STEPS_6_HOURS:
            raise ValueError("Need at least 36 rows (6h) before inference.")

        X_raw = engineered_df[self.feature_cols].tail(STEPS_6_HOURS)
        X_scaled = self.scaler_X.transform(X_raw)
        x = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_scaled = self.model(x).cpu().numpy()

        y_m = np.clip(self.scaler_y.inverse_transform(y_scaled)[0], 0, 10000)

        out = np.zeros((len(CONSOLIDATED_ZONES), len(HORIZONS)), dtype=np.float64)
        lookup = {k: v for k, v in zip(self.target_names, y_m)}
        for zi, zone in enumerate(CONSOLIDATED_ZONES):
            for hi, h in enumerate(HORIZONS):
                out[zi, hi] = float(lookup[f"target_{zone}_rvr_actual_mean_{h}"])
        return out


def _clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Sort and coerce
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 2) Interpolation strategy: short-gap linear for RVR family, general fill for met.
    rvr_like = [c for c in df.columns if "_rvr_" in c]
    met_like = [c for c in numeric_cols if c not in rvr_like]

    if rvr_like:
        df[rvr_like] = df[rvr_like].interpolate(method="linear", limit=STEPS_2_HOURS)
        df[rvr_like] = df[rvr_like].ffill().bfill()
    if met_like:
        df[met_like] = df[met_like].interpolate(method="linear", limit=STEPS_2_HOURS)
        df[met_like] = df[met_like].ffill().bfill()

    # 3) Core engineered fields
    if "temp_c" in df.columns and "dewpoint_c" in df.columns:
        dpd = (df["temp_c"] - df["dewpoint_c"]).clip(lower=0.0)
        df["dew_point_depression"] = dpd

    if "wind_dir_deg" in df.columns:
        radians = np.deg2rad(df["wind_dir_deg"] % 360.0)
        df["wind_sin"] = np.sin(radians)
        df["wind_cos"] = np.cos(radians)

    # Cyclical time
    h = df.index.hour + (df.index.minute / 60.0)
    df["hour_sin"] = np.sin(2.0 * np.pi * h / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * h / 24.0)

    m = df.index.month
    df["month_sin"] = np.sin(2.0 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * m / 12.0)

    # 4) Zone-wise lags + rolling std
    lag_feature_cols: dict[str, pd.Series] = {}
    for zone in CONSOLIDATED_ZONES:
        mean_col = f"{zone}_rvr_actual_mean"
        min_col = f"{zone}_rvr_actual_min"

        if mean_col in df.columns:
            lag_feature_cols[f"{zone}_rvr_roll_std_1h"] = df[mean_col].rolling(window=STEPS_1_HOUR, min_periods=3).std()
            lag_feature_cols[f"{zone}_rvr_mean_lag_1h"] = df[mean_col].shift(STEPS_1_HOUR)
            lag_feature_cols[f"{zone}_rvr_mean_lag_3h"] = df[mean_col].shift(STEPS_3_HOURS)
            lag_feature_cols[f"{zone}_rvr_mean_lag_6h"] = df[mean_col].shift(STEPS_6_HOURS)

        if min_col in df.columns:
            lag_feature_cols[f"{zone}_rvr_min_lag_1h"] = df[min_col].shift(STEPS_1_HOUR)
            lag_feature_cols[f"{zone}_rvr_min_lag_3h"] = df[min_col].shift(STEPS_3_HOURS)
            lag_feature_cols[f"{zone}_rvr_min_lag_6h"] = df[min_col].shift(STEPS_6_HOURS)

    if lag_feature_cols:
        lag_df = pd.DataFrame(lag_feature_cols, index=df.index)
        df = pd.concat([df, lag_df], axis=1)
        # Force a contiguous backing store to suppress fragmentation in long loops.
        df = df.copy()

    # 5) Stabilize any residual NaNs
    df = df.ffill().bfill()

    # In startup windows, long lags (e.g. 6h) can remain entirely NaN.
    # Replace any remaining non-finite values so scaler/model never receives NaNs.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def _write_map(pred_10m: dict[str, float], coords: dict[str, dict[str, float]], out_html: Path, ts: pd.Timestamp) -> None:
    fmap = folium.Map(location=[28.555, 77.095], zoom_start=14, tiles="CartoDB dark_matter")

    folium.Marker(
        [28.555, 77.095],
        popup=f"IGIA ATC | Sim cycle {ts}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(fmap)

    for zone in CONSOLIDATED_ZONES:
        if zone not in coords:
            continue
        raw_val = pred_10m.get(zone, 0.0)
        val = float(raw_val) if np.isfinite(raw_val) else 0.0
        lat = coords[zone]["lat"]
        lon = coords[zone]["lon"]
        color = _get_status_color(val)

        html = f"""
        <div style=\"text-align:center; width:120px; line-height:1.2;\">
            <div style=\"width:30px;height:30px;background:{color};border-radius:50%;display:inline-block;opacity:0.85;border:2px solid white;box-shadow:0 0 8px {color};\"></div>
            <div style=\"color:white;font-weight:bold;font-size:11pt;text-shadow:1px 1px 3px black,-1px -1px 3px black;\">{zone}<br>{int(val)}m</div>
        </div>
        """

        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=html, icon_size=(120, 70), icon_anchor=(60, 20)),
        ).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(out_html)


def _append_metric_logs(log_dir: Path, ts: pd.Timestamp, mae: float, rmse: float, r2: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "sim_mae.log").open("a", encoding="utf-8") as f:
        f.write(f"{ts.isoformat()} | MAE={mae:.4f}\n")

    with (log_dir / "sim_rmse.log").open("a", encoding="utf-8") as f:
        f.write(f"{ts.isoformat()} | RMSE={rmse:.4f}\n")

    with (log_dir / "sim_r2.log").open("a", encoding="utf-8") as f:
        f.write(f"{ts.isoformat()} | R2={r2:.6f}\n")


def run_simulation(
    input_csv: Path,
    interval_seconds: int,
    model_path: Path,
    scaler_dir: Path,
    coords_json: Path,
    map_output: Path,
    predictions_log_csv: Path,
    metrics_log_dir: Path,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input stream file not found: {input_csv}")

    df_stream = pd.read_csv(input_csv)
    if "timestamp" not in df_stream.columns:
        raise ValueError("Input file must contain a 'timestamp' column.")

    df_stream["timestamp"] = pd.to_datetime(df_stream["timestamp"])
    df_stream = df_stream.sort_values("timestamp").reset_index(drop=True)

    coords = _load_coords(coords_json)
    engine = V3RealtimeEngine(model_path=model_path, scaler_dir=scaler_dir)

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    prediction_rows: list[dict[str, float | str]] = []

    logger.info("Starting pseudo-realtime simulation loop...")
    logger.info("Cycles: %d | Interval seconds: %d", len(df_stream), interval_seconds)

    for i in range(len(df_stream)):
        current_ts = pd.Timestamp(df_stream.loc[i, "timestamp"])
        window_raw = df_stream.iloc[: i + 1].copy()
        window_features = _clean_and_engineer(window_raw)

        if len(window_features) < STEPS_6_HOURS:
            logger.info("[%s] Warm-up: %d/36 rows", current_ts, len(window_features))
            if interval_seconds > 0 and i < len(df_stream) - 1:
                time.sleep(interval_seconds)
            continue

        preds = engine.predict(window_features)
        preds = np.nan_to_num(preds, nan=0.0, posinf=10000.0, neginf=0.0)
        pred_10m = {zone: float(preds[z_idx, 0]) for z_idx, zone in enumerate(CONSOLIDATED_ZONES)}

        _write_map(pred_10m, coords, map_output, current_ts)

        # Evaluate 10-minute horizon if next timestep exists.
        if i + 1 < len(df_stream):
            truth_row = df_stream.iloc[i + 1]
            y_true_step = [float(truth_row[f"{z}_rvr_actual_mean"]) for z in CONSOLIDATED_ZONES]
            y_pred_step = [pred_10m[z] for z in CONSOLIDATED_ZONES]

            # Defensive filtering: metrics must operate only on finite pairs.
            finite_pairs = [(t, p) for t, p in zip(y_true_step, y_pred_step) if np.isfinite(t) and np.isfinite(p)]
            if not finite_pairs:
                logger.warning("[%s] No finite truth/prediction pairs for metrics; skipping this cycle.", current_ts)
                if interval_seconds > 0 and i < len(df_stream) - 1:
                    time.sleep(interval_seconds)
                continue
            y_true_step, y_pred_step = zip(*finite_pairs)

            y_true_all.extend(y_true_step)
            y_pred_all.extend(y_pred_step)

            mae = float(mean_absolute_error(y_true_all, y_pred_all))
            rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
            r2 = float(r2_score(y_true_all, y_pred_all)) if len(y_true_all) > 1 else float("nan")

            _append_metric_logs(metrics_log_dir, current_ts, mae, rmse, r2)

            logger.info(
                "[%s] Predicted +10m | Running metrics -> MAE: %.2f, RMSE: %.2f, R2: %.4f",
                current_ts,
                mae,
                rmse,
                r2,
            )

        out_row: dict[str, float | str] = {"timestamp": current_ts.isoformat()}
        out_row.update({f"pred_10m_{z}": v for z, v in pred_10m.items()})
        prediction_rows.append(out_row)

        if interval_seconds > 0 and i < len(df_stream) - 1:
            time.sleep(interval_seconds)

    predictions_log_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(predictions_log_csv, index=False)
    logger.info("Simulation complete. Predictions log saved: %s", predictions_log_csv)
    logger.info("Map output (latest cycle): %s", map_output)
    logger.info("Metric logs: %s", metrics_log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate realtime inference from synthetic airport stream file.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "data" / "interim" / "synthetic_airport_stream.csv",
        help="Synthetic stream CSV generated by generate_synthetic_airport_stream.py",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=600,
        help="Loop interval seconds (600 = 10 minutes, 0 = run without waiting)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "best_lstm_v3.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--scaler-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "scalers_v3",
        help="Directory containing scaler_X.pkl and scaler_y.pkl",
    )
    parser.add_argument(
        "--coords-json",
        type=Path,
        default=ROOT / "data" / "interim" / "sensor_coordinates.json",
        help="Zone coordinates JSON for map rendering",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=ROOT / "logs" / "igia_rvr_dashboard_simulated.html",
        help="HTML output map path",
    )
    parser.add_argument(
        "--predictions-log",
        type=Path,
        default=ROOT / "logs" / "sim_predictions_10m.csv",
        help="Per-cycle predictions CSV log",
    )
    parser.add_argument(
        "--metrics-log-dir",
        type=Path,
        default=ROOT / "logs",
        help="Directory for sim_mae.log, sim_rmse.log, sim_r2.log",
    )
    args = parser.parse_args()

    _setup_logging(args.metrics_log_dir)
    run_simulation(
        input_csv=args.input_csv,
        interval_seconds=args.interval_seconds,
        model_path=args.model_path,
        scaler_dir=args.scaler_dir,
        coords_json=args.coords_json,
        map_output=args.map_output,
        predictions_log_csv=args.predictions_log,
        metrics_log_dir=args.metrics_log_dir,
    )


if __name__ == "__main__":
    main()
