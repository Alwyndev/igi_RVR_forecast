# Deployment Guide — IGIA RVR Forecasting System

This document covers the full deployment pipeline: from containerized backend to mobile client.

---

## 1. System Architecture

```text
┌─────────────────────┐     HTTPS/JSON      ┌──────────────────────────────┐
│   Flutter Mobile     │ ◄──────────────────► │   Google Cloud Run           │
│   (Android APK)      │    /predictions_multi│                              │
│                      │                      │   Flask + Gunicorn (app.py)  │
│   • Dark-mode map    │                      │   ├── /health                │
│   • Horizon slider   │                      │   ├── /forecast (POST)       │
│   • ICAO markers     │                      │   ├── /predictions_multi     │
└─────────────────────┘                      │   └── /map                   │
                                              │                              │
                                              │   Background Task (10m)      │
                                              │   ├── Playwright Scraper     │
                                              │   ├── Preprocessor (104 F)   │
                                              │   └── MultiHorizonEngine     │
                                              └──────────────────────────────┘
```

### Production URL

```text
https://igi-rvr-api-969804968558.asia-south1.run.app
```

---

## 2. API Reference

### `GET /health`

Readiness probe. Returns model identifier.

```bash
curl https://igi-rvr-api-969804968558.asia-south1.run.app/health
```

```json
{"model": "V3.1+V5 Dynamic Hybrid", "status": "ready"}
```

---

### `POST /forecast`

Submit a 36-timestep feature window (6 hours at 10-minute resolution) and receive multi-zone, multi-horizon predictions.

**Request:**

```bash
curl -X POST \
  https://igi-rvr-api-969804968558.asia-south1.run.app/forecast \
  -H "Content-Type: application/json" \
  -d '{"features": [<36 rows of 104-feature dicts>]}'
```

**Response (200):**

```json
{
  "forecast_horizons": ["10m", "30m", "1h", "3h", "6h"],
  "zones": [
    {
      "id": "09_TDZ",
      "lat": 28.5696,
      "lon": 77.0907,
      "predictions": {"10m": 1842.3, "30m": 1790.1, "1h": 1650.5, "3h": 1200.0, "6h": 800.2}
    }
  ],
  "units": "metres"
}
```

**Error (400):** `{"error": "Exactly 36 timesteps (6 hours at 10-min) required"}`

---

### `GET /predictions_multi`

Returns the latest predictions computed from the most recent data window in the training dataset. This is the endpoint consumed by the Flutter app.

```bash
curl https://igi-rvr-api-969804968558.asia-south1.run.app/predictions_multi
```

**Response (200):**
```json
{
  "zones": [
    {"id": "09_TDZ", "lat": 28.5696, "lon": 77.0907, "predictions": {"10m": 1842, ...}},
    {"id": "10_TDZ", "lat": 28.5655, "lon": 77.0878, "predictions": {"10m": 1900, ...}}
  ],
  "horizons": ["10m", "30m", "1h", "3h", "6h"],
  "generated_at": "2026-05-09T12:00:00"
}
```

---

### `GET /map`

Serves the interactive Folium HTML dashboard. Append `?regenerate=1` to force a fresh rebuild.

```bash
curl https://igi-rvr-api-969804968558.asia-south1.run.app/map
```

---

## 3. Google Cloud Run Deployment

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
- A GCP project with Cloud Run API enabled
- Docker (for local testing, optional)

### Deployment Command

From the project root:

```bash
gcloud run deploy igi-rvr-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

Cloud Build will:

1. Read `.gcloudignore` to determine which files to upload
2. Build the Docker image using the `Dockerfile`
3. Deploy to Cloud Run with the specified configuration

### Key Configuration Decisions

#### Why Python 3.12-slim?

`contourpy==1.3.3` (a dependency of matplotlib) fails to compile from source on Python 3.10. The 3.12-slim image includes pre-built wheels.

#### `.gcloudignore` Strategy

The `.gitignore` excludes model weights (`models/*.pt`) which is correct for git, but Cloud Run needs them. The custom `.gcloudignore` overrides this behavior:

| What | Included in deploy? | Why |
| :--- | :---: | :--- |
| `models/*.pt` (V3, V5) | ✅ Yes | Required for inference |
| `data/processed/scalers_v3/` | ✅ Yes | Required for feature scaling |
| `data/processed/*.parquet` | ✅ Yes | Required for `/predictions_multi` |
| `models/best_xgboost_multi*.joblib` | ❌ No | ~1.3 GB, not used in production API |
| `flutter_app/` | ❌ No | Separate build pipeline |
| `experiments/`, `external_models/` | ❌ No | Development only |
| `Raw/`, `Latest Data/` | ❌ No | Raw data, too large |

#### Dockerfile

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser and its OS dependencies
RUN playwright install --with-deps chromium

COPY . .
EXPOSE 5000
CMD gunicorn app:app --bind 0.0.0.0:$PORT
```

Cloud Run sets the `$PORT` environment variable automatically. Gunicorn binds to it. Note the addition of `playwright install --with-deps chromium`, which is necessary because the background scraper utilizes a headless browser to extract dynamic data from the IMD RVR portal.

#### CORS

`Flask-CORS` is configured with `CORS(app)` which sets `Access-Control-Allow-Origin: *`. This allows the Flutter app (and any other client) to call the API without cross-origin restrictions.

---

## 4. Flutter APK Build

### Prerequisites

- Flutter SDK (≥2.17.0)
- Android SDK with build tools
- JDK 17+

### Build Steps

```bash
cd flutter_app
flutter pub get
flutter build apk --release
```

Output: `flutter_app/build/app/outputs/flutter-apk/app-release.apk`

### Changing the Backend URL

Edit `lib/main.dart`:

```dart
const String backendUrl =
    'https://igi-rvr-api-969804968558.asia-south1.run.app/predictions_multi';
```

### Regenerating the App Icon

```bash
flutter pub run flutter_launcher_icons
```

Icon source: `flutter_app/assets/RVR_logo.png`

---

## 5. Local Development

### Running the Backend Locally

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the development server
python app.py
```

The server starts at `http://localhost:5000`. Test with:

```bash
curl http://localhost:5000/health
curl http://localhost:5000/predictions_multi
```


### Connecting Flutter to Local Backend

In `lib/main.dart`, temporarily change `backendUrl` to:

- **Android Emulator**: `http://10.0.2.2:5000/predictions_multi`
- **Physical Device (same Wi-Fi)**: `http://<your-machine-ip>:5000/predictions_multi`

---

## 6. Monitoring & Troubleshooting

### Health Check

```bash
curl -s https://igi-rvr-api-969804968558.asia-south1.run.app/health | python -m json.tool
```

### Common Issues

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| 503 on startup | Container crash loop — model files missing | Verify `.gcloudignore` includes `.pt` files, redeploy |
| Slow cold start (~15s) | Loading PyTorch + model weights | Expected behavior; Cloud Run keeps instances warm under traffic |
| Tile rendering warnings in Flutter | Missing `userAgentPackageName` | Already fixed: set to `com.igi.antigravity` |
| `contourpy` build failure | Python version < 3.12 | Use `python:3.12-slim` base image |

### Cloud Run Logs

```bash
gcloud run services logs read igi-rvr-api --region asia-south1 --limit 50
```
