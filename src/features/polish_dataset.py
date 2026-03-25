import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def polish_dataset(input_path: str, output_path: str):
    logger.info(f"Loading optimized dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # --- 1. Circular Encoding for Wind Direction ---
    if 'wind_dir' in df.columns:
        logger.info("Applying circular encoding to wind direction...")
        df['wind_sin'] = np.sin(np.deg2rad(df['wind_dir']))
        df['wind_cos'] = np.cos(np.deg2rad(df['wind_dir']))
        df.drop(columns=['wind_dir'], inplace=True)
    
    # --- 2. Cyclical Time Features ---
    logger.info("Adding cyclical time features (hour, month)...")
    # Extract from index (datetime)
    hours = df.index.hour
    months = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['month_sin'] = np.sin(2 * np.pi * (months-1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (months-1) / 12)
    
    # --- 3. Target Generation (6-Hour Forecast) ---
    logger.info("Generating 6-hour ahead targets (36 steps)...")
    # Forecast horizon = 6 hours = 36 * 10-minute steps
    horizon = 36
    
    # We target the 'rvr_actual_mean' for all 10 zones
    rvr_targets = [c for c in df.columns if 'rvr_actual_mean' in c and 'lag' not in c and 'std' not in c]
    
    for col in rvr_targets:
        target_name = f"target_{col}_6h"
        # Shift BACKWARD so t+36 becomes target at t
        df[target_name] = df[col].shift(-horizon)
    
    # --- 4. Final NaN Cleaning ---
    initial_len = len(df)
    # Drop rows where targets are NaN (the final 6 hours of the dataset)
    df.dropna(subset=[f"target_{rvr_targets[0]}_6h"], inplace=True)
    
    # Fill residual NaNs in features
    # Use forward-fill followed by mean imputation for stability
    df.ffill(inplace=True)
    
    # Fill remaining NaNs (where ffill failed at the start) with global means
    # Passing numeric_only=True to avoid TypeError on string/categorical columns
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # --- 5. Final Quality Audit: Drop all-NaN or Zero-Variance columns ---
    # These cause scaler and NN training failures.
    logger.info("Performing final quality audit (dropping NaN/Constant columns)...")
    
    # Drop columns that are 100% NaN
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        logger.info(f"Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        df.drop(columns=all_nan_cols, inplace=True)
        
    # Drop columns that have zero variance (constant values)
    # Using df.nunique() as a robust check
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)

    logger.info(f"Dropped {initial_len - len(df)} rows due to target shifting.")
    logger.info(f"Final dataset shape: {df.shape}")
    
    # 6. Save Final Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"PHASE 3 COMPLETE. Finalized dataset saved to {output_path}")
    
    return df.columns.tolist()

if __name__ == "__main__":
    root = r"c:\Users\alwyn\OneDrive\Desktop\IGI_Antigravity"
    inp = os.path.join(root, "data", "processed", "igia_rvr_training_dataset_optimized.parquet")
    out = os.path.join(root, "data", "processed", "igia_rvr_training_dataset_final.parquet")
    polish_dataset(inp, out)
