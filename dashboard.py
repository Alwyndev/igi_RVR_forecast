"""
dashboard.py -- Spatial RVR Visualization for IGI Airport

Generates an interactive Folium map showing the 10 runway zones 
and their predicted RVR status 6 hours ahead.
"""

import folium
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.inference import RVRInferenceEngine

# Precise coordinates extracted from MET_ANTENNA.kmz
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

def get_status_color(rvr_m):
    """ICAO Fog Categories simplified"""
    if rvr_m >= 1500: return "darkgreen"
    if rvr_m >= 550:  return "orange"
    if rvr_m >= 175:  return "red"
    return "black" # Cat III Dense Fog

def create_dashboard():
    print("Generating 6-hour forecast dashboard...")
    engine = RVRInferenceEngine()
    
    # 1. Get sample data from the final parquet
    df = pd.read_parquet(ROOT / "data" / "processed" / "igia_rvr_training_dataset_final.parquet")
    feature_cols = [c for c in df.columns if "target_" not in c and c not in df.select_dtypes(exclude=[np.number]).columns]
    
    # Take a sample window (e.g., from the test set area)
    sample_input = df[feature_cols].tail(36)
    forecasts = engine.predict(sample_input)
    
    # 2. Initialize Map (Centered correctly over physical runways)
    m = folium.Map(location=[28.555, 77.095], zoom_start=14, tiles='CartoDB dark_matter')
    
    folium.Marker(
        [28.555, 77.095], 
        popup="IGIA Air Traffic Control", 
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    # 3. Add Runway Zones
    for zone, rvr in forecasts.items():
        if zone in ZONE_COORDS:
            color = get_status_color(rvr)
            coord = ZONE_COORDS[zone]
            
            folium.CircleMarker(
                location=coord,
                radius=15,
                popup=f"<b>Zone: {zone}</b><br>Forecast RVR: {rvr}m",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)
            
            folium.map.Marker(
                coord,
                icon=folium.DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 10pt; color: white; font-weight: bold;">{zone}: {int(rvr)}m</div>',
                )
            ).add_to(m)

    # 4. Legend
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white; opacity: 0.8; padding: 10px;">
     <b>RVR Forecast (6h)</b><br>
     <i style="background:darkgreen; border: 1px solid black; width: 12px; height: 12px; display: inline-block;"></i> &gt; 1500m (Good)<br>
     <i style="background:orange; border: 1px solid black; width: 12px; height: 12px; display: inline-block;"></i> 550m-1500m (CAT I)<br>
     <i style="background:red; border: 1px solid black; width: 12px; height: 12px; display: inline-block;"></i> 175m-550m (CAT II)<br>
     <i style="background:black; border: 1px solid black; width: 12px; height: 12px; display: inline-block;"></i> &lt; 175m (CAT III)<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 5. Save
    out_path = ROOT / "logs" / "igia_rvr_dashboard.html"
    m.save(out_path)
    print(f"Dashboard saved to: {out_path}")

if __name__ == "__main__":
    create_dashboard()
