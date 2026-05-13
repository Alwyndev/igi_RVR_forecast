"""
app.py -- Flask API for IGIA RVR Forecasting

Endpoint: /forecast (POST)
Payload: JSON containing 6 hours of feature data.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from src.models.inference import RVRInferenceEngine
import pandas as pd
import os

# Import MultiHorizonEngine to compute multi-horizon predictions
from dashboard_multi import MultiHorizonEngine, ZONE_COORDS, HORIZONS, create_multi_dashboard

app = Flask(__name__)
CORS(app)
engine = RVRInferenceEngine()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready", "model": "BiLSTM-V1.1-Residual"})

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        # Convert incoming JSON to DataFrame
        df = pd.DataFrame(data['features'])
        
        if len(df) != 36:
            return jsonify({"error": "Exactly 36 timesteps (6 hours at 10-min) required"}), 400
            
        predictions = engine.predict(df)
        return jsonify({
            "forecast_horizon": "6 hours",
            "predictions": predictions,
            "units": "metres"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predictions_multi', methods=['GET'])
def predictions_multi():
    """Return latest multi-horizon predictions as JSON for Flutter/native clients.

    Response format:
    {
      "zones": [
         {"id":"09_TDZ","lat":28.56,"lon":77.09,"predictions":{"10m":1200,"30m":1300,...}},
         ...
      ],
      "horizons": ["10m","30m","1h","3h","6h"],
      "generated_at": "2026-05-09T12:00:00"
    }
    """
    try:
        # Initialize engine (may be heavy; keep it simple for now)
        engine = MultiHorizonEngine()

        # Load recent processed dataset used by the engine
        ROOT = os.path.dirname(__file__)
        df_path = os.path.join(ROOT, 'data', 'processed', 'igia_rvr_training_dataset_multi.parquet')
        if not os.path.exists(df_path):
            return jsonify({"error": f"Data not found: {df_path}"}), 500

        df = pd.read_parquet(df_path)
        # Let the engine pick the relevant tail internally
        preds = engine.predict_multi(df)

        zones = []
        # Use the same canonical ordering as the engine expects
        # ZONE_COORDS keys are used here (sorted to make deterministic)
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
