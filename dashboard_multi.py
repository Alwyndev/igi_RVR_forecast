"""
dashboard_multi.py -- Interactive Multi-Horizon Time-Slider Dashboard

Uses the newly trained Multi-Horizon BiLSTM (Phase 6) to gather 
predictions across 10m, 30m, 1h, 3h, and 6h horizons.
Folium's TimestampedGeoJson plugin provides a time-slider to visualize
the progression of fog across the airport.
"""

import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
import os
import sys
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.model_v3 import RVRAttentionLSTM_V3
from src.data.runway_config import CONSOLIDATED_ZONES

# Precise coordinates from sensor_coordinates.json
ZONE_COORDS = {
    "09_TDZ":   [28.5696, 77.0907],
    "27_TDZ":   [28.5710, 77.1125],
    "10_TDZ":   [28.5655, 77.0878],
    "28_TDZ":   [28.5579, 77.1207],
    "MID_2810": [28.5628, 77.1081],
    "11_TDZ":   [28.5493, 77.0773],
    "11_BEG":   [28.5488, 77.0707],
    "29_TDZ":   [28.5436, 77.1029],
    "29_BEG":   [28.5407, 77.1081],
    "MID_2911": [28.5435, 77.0865],
}

HORIZONS = ["10m", "30m", "1h", "3h", "6h"]
HORIZON_MINUTES = [10, 30, 60, 180, 360]

def get_status_color(rvr_m):
    """ICAO Fog Category hex colors"""
    if rvr_m >= 1500: return "#006400" # darkgreen
    if rvr_m >= 550:  return "#FFA500" # orange
    if rvr_m >= 175:  return "#FF0000" # red
    return "#000000" # black

class MultiHorizonEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = ROOT / "models" / "best_lstm_v3.pt"
        scaler_dir = ROOT / "data" / "processed" / "scalers_v3"
        
        self.scaler_X = joblib.load(scaler_dir / "scaler_X.pkl")
        self.scaler_y = joblib.load(scaler_dir / "scaler_y.pkl")
        
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = RVRAttentionLSTM_V3(
            input_size=104, 
            hidden_size=384,
            num_layers=3,
            output_size=50,
            dropout=0.3
        ).to(self.device)
        
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Define canonical target ordering to match neurons 1:1
        # V3 standard: alphabetical by target column name
        self.target_names = sorted([
            f"target_{z}_rvr_actual_mean_{h}" 
            for z in CONSOLIDATED_ZONES for h in HORIZONS
        ])

    def predict_multi(self, input_features_df):
        """Returns predictions shape (10_zones, 5_horizons) -> metres"""
        # Ensure we only use numeric features and in correct order from scaler
        feature_cols = self.scaler_X.feature_names_in_
        X_scaled = self.scaler_X.transform(input_features_df[feature_cols].tail(36))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).cpu().numpy()
            
        preds_m = self.scaler_y.inverse_transform(preds_scaled)[0]
        preds_m = np.clip(preds_m, 0, 10000)
        
        # Create a lookup for alphabetical neuron mapping
        results = np.zeros((10, 5))
        neuron_lookup = {name: val for name, val in zip(self.target_names, preds_m)}
        
        for z_idx, zone in enumerate(CONSOLIDATED_ZONES):
            for h_idx, horizon in enumerate(HORIZONS):
                key = f"target_{zone}_rvr_actual_mean_{horizon}"
                results[z_idx, h_idx] = neuron_lookup[key]
                
        return results

def create_multi_dashboard():
    print("Initializing Multi-Horizon Engine...")
    engine = MultiHorizonEngine()
    
    # 1. Provide recent data from test set
    df = pd.read_parquet(ROOT / "data" / "processed" / "igia_rvr_training_dataset_multi.parquet")
    sample_input = df.tail(100) # Give it some buffer
    
    # preds matrix: shape (10 zones, 5 horizons)
    preds = engine.predict_multi(sample_input)
    
    # 2. Base Map
    m = folium.Map(location=[28.555, 77.095], zoom_start=14, tiles='CartoDB dark_matter')
    
    # ATC Marker
    folium.Marker(
        [28.555, 77.095], popup="IGIA Air Traffic Control", 
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    # 3. Create all 50 markers with hidden/visible CSS classes
    for z_idx, zone in enumerate(CONSOLIDATED_ZONES):
        if zone not in ZONE_COORDS: continue
        lat, lon = ZONE_COORDS[zone]
        
        for h_idx, h_label in enumerate(HORIZONS):
            rvr_m = float(preds[z_idx, h_idx])
            color = get_status_color(rvr_m)
            
            # The first horizon (10m) is visible by default ('block'), others hidden ('none')
            display_style = 'block' if h_idx == 0 else 'none'
            
            html = f"""
            <div class="rvr-marker rvr-horizon-{h_idx}" style="display: {display_style}; text-align: center; width: 120px; line-height: 1.2;">
                <div style="width: 30px; height: 30px; background-color: {color}; border-radius: 50%; display: inline-block; opacity: 0.8; border: 2px solid white; box-shadow: 0 0 8px {color};"></div>
                <div style="color: white; font-weight: bold; font-size: 11pt; text-shadow: 1px 1px 3px black, -1px -1px 3px black;">
                    {zone}<br>{int(rvr_m)}m
                </div>
            </div>
            """
            
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=html, 
                    icon_size=(120, 70), 
                    icon_anchor=(60, 20)
                )
            ).add_to(m)

    # 4. Inject Custom Slider UI and Logic
    slider_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999; background: rgba(30, 30, 30, 0.9); padding: 15px; border-radius: 8px; border: 2px solid #555; color: white; width: 250px; font-family: sans-serif;">
        <h3 style="margin-top: 0; margin-bottom: 15px; text-align: center;">Forecast: <span id="slider-val" style="color: #00ffcc;">10m</span></h3>
        <input type="range" min="0" max="4" value="0" id="time-slider" style="width: 100%; cursor: pointer;">
        
        <div style="margin-top: 15px; font-size: 13px;">
             <b>Visibility Legend</b><hr style="border-color: #555; margin: 5px 0;">
             <i style="background:#006400; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> &gt; 1500m (Clear)<br>
             <i style="background:#FFA500; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 550m-1500m (CAT I)<br>
             <i style="background:#FF0000; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 175m-550m (CAT II)<br>
             <i style="background:#000000; border: 1px solid white; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> &lt; 175m (CAT III Fog)<br>
         </div>
    </div>

    <script>
        var horizons = ["10m", "30m", "1h", "3h", "6h"];
        document.getElementById('time-slider').addEventListener('input', function(e) {
            var horizonIdx = e.target.value;
            // Update Text
            document.getElementById('slider-val').innerText = horizons[horizonIdx];
            
            // Hide all markers safely
            var allMarkers = document.getElementsByClassName('rvr-marker');
            for(var i=0; i<allMarkers.length; i++) {
                allMarkers[i].style.display = 'none';
            }
            
            // Show target markers
            var targetMarkers = document.getElementsByClassName('rvr-horizon-' + horizonIdx);
            for(var i=0; i<targetMarkers.length; i++) {
                targetMarkers[i].style.display = 'block';
            }
        });
    </script>
    '''
    m.get_root().html.add_child(folium.Element(slider_html))

    # 5. Save the dashboard
    out_path = ROOT / "logs" / "igia_rvr_dashboard_multi.html"
    m.save(out_path)
    print(f"Multi-Horizon Dashboard saved to: {out_path}")

if __name__ == "__main__":
    create_multi_dashboard()
