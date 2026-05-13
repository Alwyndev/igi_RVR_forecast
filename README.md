# ✈️ IGIA RVR Forecasting System
### *Multi-Horizon Runway Visual Range Prediction for Indira Gandhi International Airport*

![Champion MAE](https://img.shields.io/badge/MAE-127.51m-brightgreen?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Acc%40200m-85.75%25-blue?style=for-the-badge)
![Live](https://img.shields.io/badge/Cloud_Run-Live-4285F4?style=for-the-badge&logo=googlecloud)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch)
![Flutter](https://img.shields.io/badge/Flutter-Mobile_App-02569B?style=for-the-badge&logo=flutter)

---

## 🚀 Overview

A production-deployed RVR (Runway Visual Range) forecasting system built for IGIA's winter fog season. The system predicts visibility across **10 runway zones** at **5 time horizons** (+10m, +30m, +1h, +3h, +6h) using a **V3.1 + V5 Dynamic Hybrid** ensemble of Residual Attention LSTMs.

The backend runs on **Google Cloud Run** and serves a **Flutter mobile app** that renders predictions on an interactive dark-mode airfield map.

### Key Achievements

| Metric | Value |
|:---|:---|
| **Champion MAE** | **127.51m** (V3.1 Residual Attention LSTM) |
| **Accuracy @ 100m** | 80.73% |
| **Accuracy @ 200m** | 85.75% |
| **Fog Precision** | 83.38% (@ 600m threshold) |
| **Fog Recall** | 29.82% (Dynamic Hybrid) |
| **Baseline Beaten** | 301m → 127m (58% reduction) |

---

## 🏗️ Architecture

```
┌──────────────┐         ┌─────────────────────────┐         ┌────────────────┐
│  Sensor Data │         │   Google Cloud Run       │         │  Flutter App   │
│  (RVR/METAR/ │────────►│                          │◄────────│  (Android)     │
│   AQI feeds) │         │   Flask API (app.py)     │  HTTPS  │                │
└──────────────┘         │   ├── /health            │         │  • Dark map    │
                         │   ├── /forecast           │         │  • Horizon     │
                         │   ├── /predictions_multi  │         │    slider      │
                         │   └── /map                │         │  • ICAO pills  │
                         │                          │         └────────────────┘
                         │   V3.1 + V5 Dynamic      │
                         │   Hybrid Inference        │
                         └─────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|:---|:---|
| **Deep Learning** | PyTorch 2.11, Residual Attention LSTM (3.57M params) |
| **Ensemble** | V3.1 + V5 Dynamic Hybrid (risk-aware fog blending) |
| **Baseline** | XGBoost multi-target (50 regressors, benchmarking only) |
| **API** | Flask + Gunicorn, Flask-CORS |
| **Cloud** | Google Cloud Run (asia-south1), Python 3.12-slim |
| **Mobile** | Flutter (Dart), flutter_map, CartoDB Dark Matter tiles |
| **Data** | 104-feature pipeline (RVR, METAR, AQI, Haversine spatial) |
| **Visualization** | Folium interactive map with custom time-slider |

---

## 📦 Project Structure

```
IGI_Antigravity/
├── app.py                          # Flask API (Cloud Run entrypoint)
├── dashboard_multi.py              # Multi-horizon Folium dashboard + inference engine
├── Dockerfile                      # Container definition (Python 3.12-slim + gunicorn)
├── requirements.txt                # Python dependencies
├── Procfile                        # Gunicorn process definition
│
├── flutter_app/                    # Flutter mobile client
│   ├── lib/main.dart               #   App source (map, markers, slider)
│   ├── pubspec.yaml                #   Dependencies & icon config
│   └── assets/RVR_logo.png         #   Launcher icon source
│
├── src/
│   ├── data/                       # Data loaders & runway config
│   │   ├── build_dataset.py        #   Feature engineering pipeline
│   │   ├── runway_config.py        #   10-zone canonical ordering
│   │   ├── metar_parser.py         #   METAR/ASOS weather parsing
│   │   ├── aqi_loader.py           #   Air quality index integration
│   │   └── rvr_parser.py           #   Raw RVR data ingestion
│   │
│   └── models/                     # Model architectures & training
│       ├── model_v3.py             #   V3.1 Residual Attention LSTM (champion)
│       ├── train_v3.py             #   V3.1 training script
│       ├── train_v5.py             #   V5 asymmetric-loss variant
│       ├── train_xgboost.py        #   XGBoost baseline trainer
│       ├── inference.py            #   V1.1 single-model inference (legacy)
│       ├── realtime_pipeline.py    #   Operational 10-min polling orchestrator
│       └── benchmark_*.py          #   Comparative evaluation scripts
│
├── models/                         # Trained model weights (.pt)
│   ├── best_lstm_v3.pt             #   V3.1 champion checkpoint
│   ├── best_lstm_v5.pt             #   V5 safety-first checkpoint
│   └── best_xgboost_multi*.joblib  #   XGBoost baselines (git-ignored)
│
├── data/processed/                 # Processed datasets & scalers
│   ├── igia_rvr_training_dataset_multi.parquet
│   └── scalers_v3/                 #   StandardScaler for V3/V5
│
├── Documentation.md                # Technical research documentation
├── DEPLOYMENT.md                   # Cloud Run & Flutter deployment guide
├── PROJECT_JOURNAL.md              # Experiment log & benchmark results
├── research_abstract.md            # Research paper abstract
└── research paper.md               # Full research paper draft
```

---

## 🚦 Getting Started

### 1. Local Backend Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Test at `http://localhost:5000/health`.

### 2. Cloud Deployment

```bash
gcloud run deploy igi-rvr-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for full deployment guide, API reference, and troubleshooting.

### 3. Flutter Mobile App

```bash
cd flutter_app
flutter pub get
flutter run                    # Debug
flutter build apk --release    # Production APK
```

See **[flutter_app/README.md](flutter_app/README.md)** for configuration details.

---

## 📊 Performance — Champion Model (V3.1)

### Overall Test Metrics (2025)

| Metric | V3.1 (Champion) | External Benchmark | Improvement |
|:---|:---:|:---:|:---:|
| **MAE** | **127.51m** | 300.98m | **-57.6%** |
| **RMSE** | **370.65m** | 563.88m | **-34.3%** |
| **R²** | **0.4800** | — | — |
| **Acc @ 100m** | **80.73%** | — | — |
| **Acc @ 200m** | **85.75%** | 65.45% | **+20.3 pts** |

### Safety Trade-off (Dynamic Hybrid)

| Strategy | MAE | Fog Precision | Fog Recall | Fog F1 |
|:---|:---:|:---:|:---:|:---:|
| **V3.1 (Standard)** | 127.51m | **83.38%** | 27.09% | 0.409 |
| **V5 (Safety-first)** | 141.61m | 70.05% | **32.85%** | **0.447** |
| **Dynamic Hybrid** | **127.23m** | 78.98% | 29.82% | 0.433 |

The Dynamic Hybrid is the **production default** — it blends V3.1 and V5 with risk-aware weighting that increases V5's influence as predicted visibility drops into fog bands.

---

## 📚 Documentation

| Document | Contents |
|:---|:---|
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Cloud Run architecture, API reference, Flutter build, troubleshooting |
| **[Documentation.md](Documentation.md)** | Technical architecture, model specs, all benchmark results |
| **[PROJECT_JOURNAL.md](PROJECT_JOURNAL.md)** | Chronological experiment log with ablation studies |
| **[research_abstract.md](research_abstract.md)** | Research paper abstract |

---

*Developed for Indira Gandhi International Airport (IGIA) Flight Operations.*
