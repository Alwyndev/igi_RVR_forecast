"""
app.py -- Flask API for IGIA RVR Forecasting

Endpoint: /forecast (POST)
Payload: JSON containing 6 hours of feature data.
"""

from flask import Flask, request, jsonify
from src.models.inference import RVRInferenceEngine
import pandas as pd
import os

app = Flask(__name__)
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
