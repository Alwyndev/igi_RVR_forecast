import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_dataset(input_path: str, output_path: str):
    logger.info(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    initial_cols = len(df.columns)
    
    # 1. Define patterns to REMOVE
    remove_patterns = [
        '_limited',      # Redundant with _actual
        'mor_',          # Redundant with RVR during fog
        'voltage',       # Diagnostic only
        'blm',           # Raw telemetry
        'transmissivity',# Raw telemetry
        'valid_count',   # Metadata
        '_mor_',         # MOR variations
        'weather_str',   # Categorical (handle in Phase 4 if needed)
        'cloud_str'      # Categorical
    ]
    
    cols_to_drop = [c for c in df.columns if any(p in c for p in remove_patterns)]
    
    # Special case: Keep specific METAR features that are actually useful
    # Even if they have 'visibility' in name, we want it for ASOS
    # But RVR metrics already cover visibility. 
    # Let's keep ASOS visibility as a general check.
    
    # 2. Define features to KEEP EXPLICITLY
    # (Just in case they match remove patterns, though unlikely here)
    # We want to ensure all 10 zones have their actual_mean and actual_min
    
    df_optimized = df.drop(columns=cols_to_drop)
    
    # 3. Final Column Cleanup
    # Ensure RVR actual mean/min/lags/std are preserved
    final_cols = df_optimized.columns.tolist()
    
    logger.info(f"Dropped {len(cols_to_drop)} columns.")
    logger.info(f"Final column count: {len(final_cols)}")
    
    # Save optimized dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_optimized.to_parquet(output_path)
    logger.info(f"Saved optimized dataset to {output_path}")
    
    return final_cols

if __name__ == "__main__":
    import sys
    root = r"c:\Users\alwyn\OneDrive\Desktop\IGI_RVR_Forecast"
    inp = os.path.join(root, "data", "processed", "igia_rvr_training_dataset.parquet")
    out = os.path.join(root, "data", "processed", "igia_rvr_training_dataset_optimized.parquet")
    optimize_dataset(inp, out)
