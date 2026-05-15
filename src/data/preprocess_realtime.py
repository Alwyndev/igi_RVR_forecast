"""
preprocess_realtime.py — Real-Time Feature Engineering & Inference Runner

Takes the raw observation buffer produced by scrape_realtime.py and transforms
it into the exact 104-feature, 36-timestep input tensor that the V3/V5 Dynamic
Hybrid model expects.

Pipeline:
    1. Load raw buffer from data/realtime/latest_buffer.parquet
    2. Resample to strict 10-minute intervals
    3. Derive all 104 engineered features (lags, rolling stats, cyclical, etc.)
    4. Align columns to scaler_X.feature_names_in_ ordering
    5. Fill residual NaNs with training-set means from the fitted scaler
    6. Output a model-ready DataFrame / run inference directly

Usage:
    python -m src.data.preprocess_realtime              # preprocess only
    python -m src.data.preprocess_realtime --predict     # preprocess + run inference
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUFFER_PATH = ROOT / "data" / "realtime" / "latest_buffer.parquet"
OUTPUT_PATH = ROOT / "data" / "realtime" / "model_input.parquet"
SCALER_DIR  = ROOT / "data" / "processed" / "scalers_v3"

IST = timezone(timedelta(hours=5, minutes=30))

# Zones the model operates on (10 consolidated positions)
CONSOLIDATED_ZONES = [
    "09_TDZ", "27_TDZ", "10_TDZ", "28_TDZ", "MID_2810",
    "11_TDZ", "11_BEG", "29_TDZ", "29_BEG", "MID_2911",
]

# Timestep constants (10-minute intervals)
STEPS_1H = 6
STEPS_3H = 18
STEPS_6H = 36
SEQ_LEN  = 36    # 6-hour lookback window the model expects

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_realtime")


# ---------------------------------------------------------------------------
# Load scaler metadata
# ---------------------------------------------------------------------------
def _load_scaler_metadata():
    """Load the fitted scaler to get canonical feature order and training means."""
    scaler_path = SCALER_DIR / "scaler_X.pkl"
    if not scaler_path.exists():
        logger.error("scaler_X.pkl not found at %s", scaler_path)
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    feature_names = list(scaler.feature_names_in_)
    # Training means for each feature (used as fallback for NaNs)
    training_means = dict(zip(feature_names, scaler.mean_))
    return feature_names, training_means, scaler


# ---------------------------------------------------------------------------
# Step 1: Load and validate raw buffer
# ---------------------------------------------------------------------------
def load_buffer(path: Path = BUFFER_PATH) -> pd.DataFrame:
    """Load the raw scraped buffer and perform basic validation."""
    if not path.exists():
        logger.error("Buffer file not found: %s", path)
        logger.error("Run scrape_realtime.py first to populate the buffer.")
        sys.exit(1)

    df = pd.read_parquet(path)
    logger.info("Loaded buffer: %d rows × %d cols, range %s → %s",
                len(df), len(df.columns),
                df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# Step 2: Resample to strict 10-minute intervals
# ---------------------------------------------------------------------------
def resample_10min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to strict 10-minute grid with linear interpolation for gaps ≤2h."""
    logger.info("Resampling to 10-minute intervals...")
    df = df.resample("10min").mean()   # Average any sub-10min data
    # Linear interpolation, max 12-step gap (2 hours)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=12)
    logger.info("After resampling: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 3: Derive meteorological features
# ---------------------------------------------------------------------------
def derive_met_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw scraped columns to the canonical feature names the model expects.
    """
    logger.info("Deriving meteorological features...")

    # --- Wind speed: m/s → knots ---
    if "wind_speed_ms" in df.columns:
        df["wind_speed_kt"] = df["wind_speed_ms"] * 1.94384
    elif "wind_speed_kt" not in df.columns:
        logger.warning("No wind speed data available — filling with NaN")
        df["wind_speed_kt"] = np.nan

    # --- Visibility ---
    # visibility_m should already exist from the scraper
    if "visibility_m" not in df.columns:
        logger.warning("No visibility data — filling with NaN")
        df["visibility_m"] = np.nan

    # --- Temperature ---
    if "temp_c" not in df.columns:
        logger.warning("No temperature data — filling with NaN")
        df["temp_c"] = np.nan

    # --- Dewpoint: derive from temp + RH using Magnus formula ---
    if "dewpoint_c" not in df.columns:
        if "temp_c" in df.columns and "rh_pct" in df.columns:
            logger.info("  Computing dewpoint from T + RH (Magnus formula)")
            a, b = 17.27, 237.7
            T = df["temp_c"]
            RH = df["rh_pct"].clip(1, 100)  # Avoid log(0)
            gamma = (a * T) / (b + T) + np.log(RH / 100.0)
            df["dewpoint_c"] = (b * gamma) / (a - gamma)
        else:
            logger.warning("Cannot compute dewpoint — no T or RH data")
            df["dewpoint_c"] = np.nan

    # --- Pressure ---
    if "pressure_hpa" in df.columns and "qnh_hpa" not in df.columns:
        df["qnh_hpa"] = df["pressure_hpa"]
    elif "qnh_hpa" not in df.columns:
        logger.warning("No pressure data — filling with NaN")
        df["qnh_hpa"] = np.nan

    # --- PM2.5 / PM10: not available from WiFEX, use fallback ---
    if "pm25" not in df.columns:
        df["pm25"] = np.nan  # Will be filled with training mean later
    if "pm10" not in df.columns:
        df["pm10"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Step 4: RVR feature engineering
# ---------------------------------------------------------------------------
def engineer_rvr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the RVR-derived features the model expects:
        - {zone}_rvr_actual_min  (approximated as = mean for real-time)
        - {zone}_rvr_roll_std_1h
        - {zone}_rvr_mean_lag_{1h,3h,6h}
        - {zone}_rvr_min_lag_{1h,3h,6h}
    """
    logger.info("Engineering RVR features for %d zones...", len(CONSOLIDATED_ZONES))

    for zone in CONSOLIDATED_ZONES:
        mean_col = f"{zone}_rvr_actual_mean"
        min_col  = f"{zone}_rvr_actual_min"

        # Ensure the mean column exists
        if mean_col not in df.columns:
            logger.warning("  %s: no mean data — filling with NaN", zone)
            df[mean_col] = np.nan

        # Approximate rvr_actual_min ≈ rvr_actual_mean (instantaneous min unavailable)
        df[min_col] = df[mean_col]

        # Rolling standard deviation (1 hour = 6 steps)
        df[f"{zone}_rvr_roll_std_1h"] = (
            df[mean_col].rolling(window=STEPS_1H, min_periods=2).std()
        )

        # Lag features
        for lag_label, lag_steps in [("1h", STEPS_1H), ("3h", STEPS_3H), ("6h", STEPS_6H)]:
            df[f"{zone}_rvr_mean_lag_{lag_label}"] = df[mean_col].shift(lag_steps)
            df[f"{zone}_rvr_min_lag_{lag_label}"]  = df[min_col].shift(lag_steps)

    return df


# ---------------------------------------------------------------------------
# Step 5: Dew point depression
# ---------------------------------------------------------------------------
def add_dew_point_depression(df: pd.DataFrame) -> pd.DataFrame:
    """dew_point_depression = temp_c - dewpoint_c, clamped ≥ 0."""
    logger.info("Computing dew point depression...")
    df["dew_point_depression"] = (df["temp_c"] - df["dewpoint_c"]).clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Step 6: Cyclical encodings
# ---------------------------------------------------------------------------
def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circular wind and temporal features."""
    logger.info("Adding cyclical feature encodings...")

    # Wind direction (circular)
    if "wind_dir" in df.columns:
        df["wind_sin"] = np.sin(np.deg2rad(df["wind_dir"]))
        df["wind_cos"] = np.cos(np.deg2rad(df["wind_dir"]))
    else:
        df["wind_sin"] = np.nan
        df["wind_cos"] = np.nan

    # Hour of day
    hours = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Month of year
    months = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12)

    return df


# ---------------------------------------------------------------------------
# Step 7: Column alignment & NaN filling
# ---------------------------------------------------------------------------
def align_and_fill(df: pd.DataFrame, feature_names: list, training_means: dict) -> pd.DataFrame:
    """
    Align DataFrame columns to the exact scaler ordering and fill NaNs.

    NaN strategy:
        1. Forward-fill  (carry last known value)
        2. Backward-fill (for leading NaNs)
        3. Training-set mean fallback (from scaler.mean_)
    """
    logger.info("Aligning to %d model features...", len(feature_names))

    # Add any missing columns
    for col in feature_names:
        if col not in df.columns:
            logger.debug("  Adding missing column: %s", col)
            df[col] = np.nan

    # Select and order
    df_aligned = df[feature_names].copy()

    # Fill NaNs: forward-fill then backward-fill
    df_aligned = df_aligned.ffill().bfill()

    # Final fallback: fill remaining NaNs with training means
    remaining_nans = df_aligned.isna().sum()
    nan_cols = remaining_nans[remaining_nans > 0]
    if len(nan_cols) > 0:
        logger.info("  Filling %d columns with training means", len(nan_cols))
        fill_values = {}
        for col in nan_cols.index:
            fill_values[col] = training_means.get(col, 0.0)
        df_aligned = df_aligned.fillna(fill_values)

    return df_aligned


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess(buffer_path: Path = BUFFER_PATH) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Returns a DataFrame with exactly 104 columns in the scaler's canonical order,
    ready to be fed into MultiHorizonEngine.predict_multi().
    """
    logger.info("=" * 60)
    logger.info("Real-Time Preprocessing Pipeline")
    logger.info("=" * 60)

    # Load scaler metadata
    feature_names, training_means, scaler = _load_scaler_metadata()
    logger.info("Model expects %d features", len(feature_names))

    # Load raw buffer
    df = load_buffer(buffer_path)

    # Resample to 10-min grid
    df = resample_10min(df)

    # Derive met features
    df = derive_met_features(df)

    # Dew point depression
    df = add_dew_point_depression(df)

    # RVR feature engineering
    df = engineer_rvr_features(df)

    # Cyclical encodings
    df = add_cyclical_features(df)

    # Align & fill
    df_model = align_and_fill(df, feature_names, training_means)

    # Verify completeness
    assert df_model.shape[1] == len(feature_names), (
        f"Column count mismatch: {df_model.shape[1]} vs {len(feature_names)}"
    )
    nan_count = df_model.isna().sum().sum()
    if nan_count > 0:
        logger.error("Still have %d NaN values after preprocessing!", nan_count)
    else:
        logger.info("[OK] Zero NaN values -- data is model-ready")

    # Save preprocessed input
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_parquet(OUTPUT_PATH)
    logger.info("Model input saved to %s (%d rows × %d cols)",
                OUTPUT_PATH, len(df_model), df_model.shape[1])

    return df_model


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------
def run_inference(df_model: pd.DataFrame):
    """
    Feed the preprocessed data into the Dynamic Hybrid model and print predictions.
    """
    logger.info("=" * 60)
    logger.info("Running V3/V5 Dynamic Hybrid Inference")
    logger.info("=" * 60)

    if len(df_model) < SEQ_LEN:
        logger.error(
            "Insufficient data: need %d timesteps (6h), got %d. "
            "Wait for more data accumulation or check scraper.",
            SEQ_LEN, len(df_model),
        )
        return None

    from dashboard_multi import MultiHorizonEngine, ZONE_COORDS, HORIZONS

    engine = MultiHorizonEngine()
    # predict_multi() internally takes .tail(36) and scales
    preds = engine.predict_multi(df_model)

    # Format results
    zone_keys = sorted(list(ZONE_COORDS.keys()))
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")

    print(f"\n{'='*70}")
    print(f"  IGIA RVR Forecast — {now_ist}")
    print(f"  Model: V3.1 + V5 Dynamic Hybrid")
    print(f"  Input window: {df_model.index[-SEQ_LEN]} -> {df_model.index[-1]}")
    print(f"{'='*70}\n")

    # Header
    horizons = HORIZONS
    header = f"  {'Zone':<12s}" + "".join(f"{'T+'+h:>10s}" for h in horizons) + "   Status"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for z_idx, zone in enumerate(zone_keys):
        row = f"  {zone:<12s}"
        min_rvr = float("inf")
        for h_idx, h in enumerate(horizons):
            rvr = float(preds[z_idx, h_idx])
            min_rvr = min(min_rvr, rvr)
            row += f"{int(rvr):>9d}m"
        # Status
        if min_rvr >= 1500:
            status = "[CLEAR]"
        elif min_rvr >= 550:
            status = "[CAT-I]"
        elif min_rvr >= 175:
            status = "[CAT-II]"
        else:
            status = "[CAT-III]"
        row += f"   {status}"
        print(row)

    print()
    return preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess real-time data for IGIA RVR model")
    parser.add_argument("--predict", action="store_true", help="Run model inference after preprocessing")
    parser.add_argument("--buffer", type=str, default=str(BUFFER_PATH), help="Path to raw buffer parquet")
    args = parser.parse_args()

    df_model = preprocess(Path(args.buffer))

    print(f"\n{'='*60}")
    print(f"  Preprocessing Summary")
    print(f"{'='*60}")
    print(f"  Rows             : {len(df_model)}")
    print(f"  Features         : {df_model.shape[1]} (expected: 104)")
    print(f"  Time range       : {df_model.index.min()} -> {df_model.index.max()}")
    print(f"  NaN remaining    : {df_model.isna().sum().sum()}")
    print(f"  Has 6h window?   : {'Yes' if len(df_model) >= SEQ_LEN else 'Need more data'}")
    print(f"  Output saved to  : {OUTPUT_PATH}")

    if args.predict:
        run_inference(df_model)
    else:
        print(f"\n  To run inference: python -m src.data.preprocess_realtime --predict")


if __name__ == "__main__":
    main()
