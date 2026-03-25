"""
build_multi_targets.py -- Option B: Expanding the dataset to support 5 sequential horizons.

This script takes the standard V1.1 dataset (which only has 6h targets) and appends
intermediate targets: 10m, 30m, 1h, and 3h for all 10 runway zones.

Input: igia_rvr_training_dataset.parquet (The pre-target base dataset)
Output: igia_rvr_training_dataset_multi.parquet (Contains 50 target columns)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Local Imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.runway_config import CONSOLIDATED_ZONES

def generate_multi_horizon_targets(input_parquet: str, output_parquet: str):
    print(f"Loading base dataset: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    # 10m = 1 step
    # 30m = 3 steps
    # 1h  = 6 steps
    # 3h  = 18 steps
    # 6h  = 36 steps
    horizons = {
        "10m": 1,
        "30m": 3,
        "1h": 6,
        "3h": 18,
        "6h": 36
    }
    
    print("Generating temporal shift targets...")
    for zone in CONSOLIDATED_ZONES:
        mean_col = f"{zone}_rvr_actual_mean"
        if mean_col in df.columns:
            for h_label, steps in horizons.items():
                target_col = f"target_{zone}_rvr_actual_mean_{h_label}"
                df[target_col] = df[mean_col].shift(-steps)
                
    # Drop rows that don't have enough future data to form a complete 6h forward sequence
    # The maximum shift is 36, so the last 36 rows will have NaNs in the 6h targets.
    # We must drop them to maintain a clean supervised dataset.
    print(f"Dataset shape before target drop: {df.shape}")
    
    # Check for target NaNs across any of the 50 columns
    target_cols = [c for c in df.columns if c.startswith("target_")]
    print(f"Total target columns generated: {len(target_cols)} (Expected: 50)")
    
    df = df.dropna(subset=target_cols)
    print(f"Dataset shape after horizon drop: {df.shape}")
    
    print(f"Saving multi-horizon dataset to: {output_parquet}")
    df.to_parquet(output_parquet)
    print("Complete.")

if __name__ == "__main__":
    in_file = ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet"        
    out_file = ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet"
    
    generate_multi_horizon_targets(str(in_file), str(out_file))
