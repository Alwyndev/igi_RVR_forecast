# IGIA RVR Predictive Modeling — Master Documentation
*A High-Precision 6-Hour Forecasting System for Indira Gandhi International Airport*

---

## 1. Executive Summary
This project delivers a multi-zone, multi-horizon RVR (Runway Visual Range) forecasting system. By integrating 5 years of heterogeneous meteorological data (RVR, METAR, AQI), we achieved a high-precision **269m MAE** target for 6-hour ahead predictions using a Residual BiLSTM architecture.

---

## 2. The Data Ecosystem

### 2.1 Sources & Resolution
- **RVR Sensor Logs (10-second):** Raw transmissometer data from 13 sensor positions.
- **METAR/ASOS (30-minute):** Canonical atmospheric data (Temp, Dew Pt, Wind, Pressure) from VIDP station.
- **AQI/CPCB (1-hour):** Particulate matter (PM2.5, PM10) which significantly affects visibility during non-foggy haze periods.
- **Physical KMZ:** `MET_ANTENNA.kmz` used to extract sub-meter accurate GPS coordinates for all 10 runway zones.

### 2.2 Data Cleaning & Resampling
- **Frequency:** Standardized to a **10-minute** interval.
- **RVR Aggregation:** Captured both `MEAN` (nominal visibility) and `MIN` (worst-case visibility) within each 10-minute window.
- **String Cleaning:** Removed non-numeric noise (e.g., `****`, `M`, `V`, `L`) and replaced with NaNs to be handled by interpolation.
- **Circular Encoding:** Wind direction (`0-360`) and Time-of-day were encoded as `sin/cos` pairs to ensure mathematical continuity (e.g., 350° and 10° are close).

---

## 3. Spatial & Temporal Feature Engineering

### 3.1 Haversine Spatial Interpolation
IGIA's runway zones are physically linked. We implemented a distance-weighted spatial interpolator that reconstructs missing sensor data using active sensors on the **same physical strip**.
- **Example:** If `28_TDZ` and `28_MID` are active but `10_TDZ` is down, we interpolate the missing value based on strip geometry.

### 3.2 Feature Selection Logic
Reduced feature space from 333 to **104 variables** based on the Pearson Correlation matrix:
- **Core Drivers:** Dew Point Depression ($T - T_d$), PM2.5, and 1-hour Rolling Standard Deviation (capturing fog instability).
- **Temporal Lags:** 1h, 3h, and 6h lags for all RVR zones were included to provide the BiLSTM with historical trend context.

---

## 4. Modeling & Research Breakthroughs

### 4.1 Architecture Evolution
| Phase | Model | Parameters | Metric (MAE) | Verdict |
|:---|:---|:---|:---|:---|
| **V1** | Stacked BiLSTM | 2.3M | 289.6m | Success (Baseline) |
| **V2** | BiLSTM + Attention | 15.6M | 773.9m | **Failed** (Overfit) |
| **V1.1** | **Residual BiLSTM** | **7.2M** | **269.1m** | **CHAMPION** ✅ |

### 4.2 The Z-Score Breakthrough (StandardScaler)
A critical finding during the project was the failure of `MinMaxScaler`. Because RVR is mostly clear (>2000m), MinMax squashed the "fog onset" gradients into a tiny range. By switching to `StandardScaler` (Z-Score), we allowed the network to treat a drop from 3000m to 300m as a "3-standard-deviation event," drastically improving loss convergence and prediction of sudden visibility drops.

---

## 5. Multi-Horizon Forecasting (Phase 6)
To support a Time-Slider UI, we built a **True Multi-Horizon** model (`model_multi.py`):
- **Output:** 50 neurons (10 Zones × 5 Horizons).
- **Horizons:** +10m, +30m, +1h, +3h, +6h.
- **Capability:** This model predicts the entire "fog curve," allowing ATC to see exactly when visibility is expected to reach CAT-II (550m) or CAT-III (175m) thresholds.

---

## 6. Operational Manual (Deployment)

### 6.1 Real-Time Operations (`realtime_pipeline.py`)
This script acts as the production orchestrator. It should be run in a persistent environment (e.g., a GPU-enabled server).
1.  **Polling:** Every 10 mins, it scans the `Latest Data/` folder.
2.  **Inference:** It picks up the last 36 steps of sensor data and runs the Multi-Horizon prediction.
3.  **Dashboard Update:** It regenerates the Folium HTML map.

### 6.2 Running the System
```powershell
# 1. Start the Flask API (for data queries)
python app.py

# 2. Start the Real-Time Update loop
python -m src.models.realtime_pipeline --interval 600

# 3. Access Dashboard
Navigate to: igia_rvr_dashboard_multi.html
```

---

## 7. Folder Structure
- `/src/data/`: Parsons and kmz extraction.
- `/src/features/`: Dataset cleaning, interpolation, and multi-horizon target building.
- `/src/models/`: Training scripts and architecture definitions.
- `/models/`: Checkpointed `.pt` files and `.pkl` scalers.
- `/logs/`: API logs and final rendered dashboards.

---
*End of Documentation*
