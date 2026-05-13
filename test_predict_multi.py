import json
import os
import pandas as pd

from dashboard_multi import MultiHorizonEngine, ZONE_COORDS, HORIZONS

ROOT = os.path.dirname(__file__)
DF_PATH = os.path.join(ROOT, 'data', 'processed', 'igia_rvr_training_dataset_multi.parquet')

if not os.path.exists(DF_PATH):
    raise SystemExit(f"Data file not found: {DF_PATH}")

print('Loading data...')
df = pd.read_parquet(DF_PATH)
print('Initializing engine (may take a moment)...')
engine = MultiHorizonEngine()
print('Running prediction...')
preds = engine.predict_multi(df)

zones = []
zone_keys = sorted(list(ZONE_COORDS.keys()))
for z_idx, zone in enumerate(zone_keys):
    lat, lon = ZONE_COORDS[zone]
    zone_preds = {h: float(preds[z_idx, h_idx]) for h_idx, h in enumerate(HORIZONS)}
    zones.append({"id": zone, "lat": lat, "lon": lon, "predictions": zone_preds})

output = {"zones": zones, "horizons": HORIZONS, "generated_at": pd.Timestamp.now().isoformat()}
print(json.dumps(output, indent=2))
