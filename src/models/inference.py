"""
inference.py -- Production Inference Engine for IGIA RVR (V1.1)

Loads:
  - best_bilstm_v1_1.pt
  - scaler_X.pkl / scaler_y.pkl (Z-Score)

Calculates 6-hour ahead forecasts for all 10 runway zones.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_v1_1 import RVRBiLSTM_V1_1
from src.models.dataset import TARGET_ZONES

class RVRInferenceEngine:
    def __init__(self, model_path=None, scaler_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        if model_path is None:
            model_path = ROOT / "models" / "best_bilstm_v1_1.pt"
        if scaler_dir is None:
            scaler_dir = ROOT / "data" / "processed" / "scalers_v1_1"
            
        # 1. Load Scalers
        self.scaler_X = joblib.load(scaler_dir / "scaler_X.pkl")
        self.scaler_y = joblib.load(scaler_dir / "scaler_y.pkl")
        
        # 2. Load Model
        ckpt = torch.load(model_path, map_location=self.device)
        self.config = ckpt['config']
        
        self.model = RVRBiLSTM_V1_1(
            input_size=104, # Verified 104 features
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=0.0 # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Inference engine ready on {self.device} (MAE: {ckpt['val_mae']:.1f}m)")

    def predict(self, input_features_df):
        """
        Expects a DataFrame with 36 timesteps (6 hours) of data.
        Returns: Dict of {zone_name: forecast_value_metres}
        """
        if len(input_features_df) < 36:
            raise ValueError("Inference requires exactly 36 timesteps (6 hours) of context.")
            
        # Scale
        X_scaled = self.scaler_X.transform(input_features_df.tail(36))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).cpu().numpy()
            
        # Inverse Scale
        preds_m = self.scaler_y.inverse_transform(preds_scaled)[0]
        preds_m = np.clip(preds_m, 0, 10000)
        
        return {zone: round(float(m), 2) for zone, m in zip(TARGET_ZONES, preds_m)}

if __name__ == "__main__":
    # Test with a snippet from the dataset
    engine = RVRInferenceEngine()
    df = pd.read_parquet(ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet")
    
    # Remove target columns for inference
    feature_cols = [c for c in df.columns if "target_" not in c and c not in df.select_dtypes(exclude=[np.number]).columns]
    sample_input = df[feature_cols].iloc[100:136]
    
    forecasts = engine.predict(sample_input)
    print("\n--- 6-Hour Forecast ---")
    for zone, val in forecasts.items():
        print(f"{zone:<15}: {val:>8.1f} m")
