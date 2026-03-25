# ✈️ IGIA RVR Predictive Modeling
### *High-Precision 6-Hour multi-zone fog forecasting for Indira Gandhi International Airport (DEL)*

![RVR Dashboard Preview](https://img.shields.io/badge/MAE-269.1m-green?style=for-the-badge)
![NVIDIA RTX 5070 Ti](https://img.shields.io/badge/Powered_by-NVIDIA_Blackwell-76B900?style=for-the-badge&logo=nvidia)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch_Nightly-EE4C2C?style=for-the-badge&logo=pytorch)

## 🚀 Overview
This repository contains a production-ready RVR (Runway Visual Range) forecasting system. Developed specifically for the winter fog challenges at IGIA, the system utilizes a **Residual BiLSTM** architecture to predict visibility across 10 critical runway zones simultaneously.

### **Key Achievements**
- **Record Accuracy:** Achieved a **269.1m Mean Absolute Error (MAE)** on the 2025 unseen test set, significantly outperforming standard persistence models.
- **Multi-Horizon Intelligence:** A single 50-neuron output head predicts visibility at **+10m, +30m, +1h, +3h, and +6h** horizons in real-time.
- **Interactive Spatial Dashboard:** A Folium-based map with a time-slider allows Air Traffic Controllers (ATC) to visualize the predicted "fog onset" path across the airport.

---

## 🛠️ The Technology Stack
- **Deep Learning:** PyTorch (nightly) with Mixed Precision (AMP) training.
- **Hardware:** Optimized for **RTX 5070 Ti (Blackwell)** Tensor Cores.
- **Data Engineering:** Automated Haversine spatial interpolation and circular wind encoding.
- **Visualization:** Interactive Folium maps with custom CSS-driven time sliders.

---

## 📦 Project Structure
- `src/models/model_multi.py`: The 50-neuron Multi-Horizon Residual BiLSTM.
- `src/models/realtime_pipeline.py`: The operational orchestrator for live sensor feeds.
- `dashboard_multi.py`: Generator for the interactive time-slider visualization.
- `Documentation.md`: Comprehensive 200+ line project manual and research findings.

---

## 🚦 Getting Started

### 1. Installation
```bash
# Recommendation: use Python 3.12+
pip install -r requirements.txt
```

### 2. Running an Operational Cycle
The system is built to poll live data every 10 minutes:
```bash
python -m src.models.realtime_pipeline --interval 600
```

### 3. Visualizing the Forecast
Open `logs/igia_rvr_dashboard_multi.html` in any browser to access the interactive slider.

---

## 📊 Performance Benchmark (V1.1 Gold)
By switching from MinMax to **Z-Score (StandardScaler)**, the model effectively captured "visibility cliffs"—sudden fog drops that standard scales often miss.

| Metric | V1 (Baseline) | V2 (Attention) | **V1.1 (Residual)** |
|:---|:---|:---|:---|
| **Test MAE** | 289.6m | 773.0m | **269.1m** |
| **Stability** | High | Low | **Ultra-High** |

---
*Developed for Indira Gandhi International Airport (IGIA) Flight Operations.*
