"""
app.py -- Flask API for IGIA RVR Forecasting (V3/V5 Dynamic Hybrid)

Endpoints:
  GET  /health             – Readiness probe
  POST /forecast            – Single-window multi-horizon RVR prediction
  GET  /predictions_multi   – Latest grouped predictions for all zones
  GET  /map                 – Folium HTML dashboard
"""

import atexit
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
from pathlib import Path

# Scheduler imports
from apscheduler.schedulers.background import BackgroundScheduler

from dashboard_multi import MultiHorizonEngine, ZONE_COORDS, HORIZONS, create_multi_dashboard

# Pipeline imports
from src.data.scrape_realtime import assemble_buffer
from src.data.preprocess_realtime import preprocess, BUFFER_PATH

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Background Scheduler Task
# ---------------------------------------------------------------------------
def run_realtime_pipeline():
    """Background task to fetch live data, preprocess it, and update the dashboard."""
    print("\n[Scheduler] Running real-time pipeline (scrape -> preprocess -> dashboard)...")
    try:
        # 1. Scrape data and update latest_buffer.parquet
        assemble_buffer()
        
        # 2. Preprocess buffer and save to model_input.parquet
        preprocess(BUFFER_PATH)
        
        # 3. Regenerate Folium map dashboard with latest data
        create_multi_dashboard()
        
        print("[Scheduler] Pipeline cycle complete.\n")
    except Exception as e:
        print(f"[Scheduler] Pipeline encountered an error: {e}\n")

# Initialize and start scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=run_realtime_pipeline, trigger="interval", minutes=10)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

# Run the pipeline once on startup so data is immediately available
# (You might want to comment this out if testing locally to speed up startup)
print("Starting initial pipeline run...")
run_realtime_pipeline()

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready", "model": "V3.1+V5 Dynamic Hybrid"})

@app.route('/forecast', methods=['POST'])
def forecast():
    """Single-window forecast using V3/V5 Dynamic Hybrid.

    Expects JSON: {"features": [<36 rows of 104-feature dicts>]}
    Returns per-zone, per-horizon predictions in metres.
    """
    try:
        data = request.json
        df = pd.DataFrame(data['features'])

        if len(df) != 36:
            return jsonify({"error": "Exactly 36 timesteps (6 hours at 10-min) required"}), 400

        engine = MultiHorizonEngine()
        preds = engine.predict_multi(df)

        zone_keys = sorted(list(ZONE_COORDS.keys()))
        zones = []
        for z_idx, zone in enumerate(zone_keys):
            lat, lon = ZONE_COORDS[zone]
            zone_preds = {h: float(preds[z_idx, h_idx]) for h_idx, h in enumerate(HORIZONS)}
            zones.append({"id": zone, "lat": lat, "lon": lon, "predictions": zone_preds})

        return jsonify({
            "forecast_horizons": HORIZONS,
            "zones": zones,
            "units": "metres"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predictions_multi', methods=['GET'])
def predictions_multi():
    """Return latest multi-horizon predictions as JSON for Flutter/native clients."""
    try:
        engine = MultiHorizonEngine()

        # Load live preprocessed dataset instead of static test set
        ROOT = os.path.dirname(__file__)
        df_path = os.path.join(ROOT, 'data', 'realtime', 'model_input.parquet')
        
        if not os.path.exists(df_path):
            return jsonify({"error": f"Live data not found: {df_path}"}), 500

        df = pd.read_parquet(df_path)
        
        # Engine takes the tail 36 timesteps automatically in predict_multi
        # but let's be explicit and pass the expected 36 timesteps (6 hours)
        sample_input = df.tail(36)
        
        preds = engine.predict_multi(sample_input)

        zones = []
        zone_keys = sorted(list(ZONE_COORDS.keys()))
        for z_idx, zone in enumerate(zone_keys):
            lat, lon = ZONE_COORDS[zone]
            zone_preds = {h: float(preds[z_idx, h_idx]) for h_idx, h in enumerate(HORIZONS)}
            zones.append({"id": zone, "lat": lat, "lon": lon, "predictions": zone_preds})

        return jsonify({"zones": zones, "horizons": HORIZONS, "generated_at": pd.Timestamp.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Simple index linking to map and API endpoints."""
    html = (
        "<html><head><title>IGI RVR API</title></head><body>"
        "<h2>IGI RVR Service</h2>"
        "<ul>"
        "<li><a href=\"/map\">Map (HTML)</a></li>"
        "<li><a href=\"/predictions_multi\">Predictions (JSON)</a></li>"
        "</ul>"
        "</body></html>"
    )
    return html


@app.route('/map', methods=['GET'])
def serve_map():
    """Serve the generated Folium HTML map. Use ?regenerate=1 to rebuild before serving."""
    try:
        regenerate = request.args.get('regenerate', '0') in ('1', 'true', 'True')
        root_dir = os.path.dirname(__file__)
        out_path = os.path.join(root_dir, 'logs', 'igia_rvr_dashboard_multi.html')

        if regenerate or not os.path.exists(out_path):
            # Recreate the dashboard on demand
            create_multi_dashboard()

        if not os.path.exists(out_path):
            return jsonify({"error": "Map file not found after generation."}), 500

        return send_file(out_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
