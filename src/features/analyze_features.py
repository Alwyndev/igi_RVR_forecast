import pandas as pd
import numpy as np
import os

def analyze_features(parquet_path: str):
    print(f"Loading dataset from {parquet_path}...")
    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return
        
    df = pd.read_parquet(parquet_path)
    
    # 1. Basic Correlation with a sample target (09_TDZ_rvr_actual_mean)
    target = '09_TDZ_rvr_actual_mean'
    if target not in df.columns:
        targets = [c for c in df.columns if 'rvr_actual_mean' in c]
        target = targets[0] if targets else None
        
    print(f"Target for correlation analysis: {target}")
    
    # Drop rows with NaN in target for correlation
    df_clean = df.dropna(subset=[target])
    
    # Select numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Compute correlation with target
    corrs = df_clean[numeric_cols].corr()[target].sort_values(ascending=False)
    
    print("\nTop 20 Positively Correlated Features with 09_TDZ RVR:")
    print(corrs.head(20))
    
    print("\nTop 20 Negatively Correlated Features with 09_TDZ RVR:")
    print(corrs.tail(20).sort_values())
    
    # 2. Identify Redundant Pairs (Pearson > 0.98)
    corr_matrix = df_clean[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    
    print(f"\nSuggesting {len(to_drop)} redundant features for removal (>0.98 correlation).")
    print("Sample Redundant Features:", to_drop[:15])
    
    # 3. Missingness
    missing = df.isnull().mean()
    high_missing = missing[missing > 0.4].index.tolist()
    print(f"\nFeatures with > 40% missing values: {len(high_missing)}")
    print(high_missing[:10])

if __name__ == "__main__":
    analyze_features(r"c:\Users\alwyn\OneDrive\Desktop\IGI_Antigravity\data\processed\igia_rvr_training_dataset.parquet")
