"""
Generate synthetic airport sensor stream for up to 24 hours in 10-minute intervals.

Output schema is intentionally aligned with the realtime simulation pipeline so it can
be cleaned, feature-engineered, and fed to the trained V3 model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.runway_config import CONSOLIDATED_ZONES

ROOT = Path(__file__).resolve().parents[2]


def _zone_baseline(zone_idx: int, n: int, rng: np.random.Generator) -> np.ndarray:
    # Zone-specific offsets avoid every runway evolving identically.
    base = 1400.0 + 180.0 * np.sin(np.linspace(0, 2 * np.pi, n) + zone_idx * 0.45)

    # Random fog episodes depress visibility for contiguous windows.
    fog_signal = np.zeros(n, dtype=np.float64)
    episodes = rng.integers(1, 4)
    for _ in range(episodes):
        center = int(rng.integers(0, n))
        width = int(rng.integers(6, 24))
        depth = float(rng.uniform(500.0, 1100.0))
        idx = np.arange(n)
        fog_signal -= depth * np.exp(-0.5 * ((idx - center) / max(width, 1)) ** 2)

    # Gentle autoregressive noise to mimic sensor continuity.
    noise = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        noise[i] = 0.86 * noise[i - 1] + rng.normal(0.0, 32.0)

    rvr_mean = base + fog_signal + noise + 650.0
    rvr_mean = np.clip(rvr_mean, 50.0, 3500.0)
    return rvr_mean


def generate_synthetic_stream(hours: int, interval_min: int, seed: int) -> pd.DataFrame:
    if hours <= 0:
        raise ValueError("hours must be > 0")
    if interval_min <= 0:
        raise ValueError("interval_min must be > 0")

    rng = np.random.default_rng(seed)
    periods = int((hours * 60) / interval_min)
    if periods < 2:
        raise ValueError("At least 2 timesteps are required.")

    end_ts = pd.Timestamp.now().floor(f"{interval_min}min")
    timestamps = pd.date_range(end=end_ts, periods=periods, freq=f"{interval_min}min")

    phase = np.linspace(0.0, 2.0 * np.pi, periods)
    temp_c = 18.0 + 9.5 * np.sin(phase - 0.75) + rng.normal(0.0, 0.5, periods)
    dewpoint_c = temp_c - (1.8 + 2.5 * (1.0 - np.sin(phase + 0.25))) + rng.normal(0.0, 0.35, periods)
    qnh_hpa = 1012.5 + 2.8 * np.sin(phase / 2.0) + rng.normal(0.0, 0.2, periods)
    wind_speed_kt = np.clip(7.0 + 4.5 * np.sin(phase + 0.8) + rng.normal(0.0, 1.2, periods), 0.0, 35.0)

    wind_dir_deg = np.zeros(periods, dtype=np.float64)
    wind_dir_deg[0] = float(rng.uniform(0.0, 360.0))
    for i in range(1, periods):
        wind_dir_deg[i] = (wind_dir_deg[i - 1] + rng.normal(0.0, 12.0)) % 360.0

    pm25 = np.clip(70.0 + 35.0 * (1.0 - np.sin(phase + 0.4)) + rng.normal(0.0, 6.0, periods), 8.0, 350.0)
    pm10 = np.clip(pm25 * 1.5 + rng.normal(0.0, 10.0, periods), 20.0, 600.0)

    data: dict[str, np.ndarray] = {
        "timestamp": timestamps,
        "temp_c": temp_c,
        "dewpoint_c": dewpoint_c,
        "qnh_hpa": qnh_hpa,
        "wind_speed_kt": wind_speed_kt,
        "wind_dir_deg": wind_dir_deg,
        "pm25": pm25,
        "pm10": pm10,
    }

    all_zone_means = []
    for z_idx, zone in enumerate(CONSOLIDATED_ZONES):
        rvr_mean = _zone_baseline(z_idx, periods, rng)
        rvr_min = np.clip(rvr_mean - np.abs(rng.normal(90.0, 40.0, periods)), 25.0, rvr_mean)
        data[f"{zone}_rvr_actual_mean"] = rvr_mean
        data[f"{zone}_rvr_actual_min"] = rvr_min
        all_zone_means.append(rvr_mean)

    stack_means = np.vstack(all_zone_means)
    visibility_m = np.clip(np.percentile(stack_means, 20, axis=0) + rng.normal(0.0, 30.0, periods), 30.0, 5000.0)
    data["visibility_m"] = visibility_m

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IGIA-like 10-minute sensor stream.")
    parser.add_argument("--hours", type=int, default=24, help="Duration to synthesize (hours)")
    parser.add_argument("--interval-min", type=int, default=10, help="Interval in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "interim" / "synthetic_airport_stream.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = generate_synthetic_stream(hours=args.hours, interval_min=args.interval_min, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved synthetic stream: {args.output}")
    print(f"Rows: {len(df)} | Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
