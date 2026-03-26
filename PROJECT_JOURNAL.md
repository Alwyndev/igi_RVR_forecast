# PROJECT JOURNAL — IGIA RVR BiLSTM Predictive Model

> **Project**: 6-Hour Ahead RVR Prediction for Indira Gandhi International Airport  
> **Model**: Unified Residual Attention LSTM (V3.1)  
> **Author**: Alwyn  
> **Started**: 2026-03-24  

---

## Change Log

| Date | File / Action | Description |
|:---|:---|:---|
| 2026-03-24 | Project Init | Created directory structure, requirements.txt, .gitignore |
| 2026-03-24 | Data Analysis | identified 5 disparate sources (RVR, ASOS, AQI, DCWIS, KMZ) |
| 2026-03-24 | Phase 2 | Unified 10-min resampling and Haversine spatial interpolation |
| 2026-03-25 | Phase 6 | True Multi-Horizon forecasting (+10m to +6h) developed |
| 2026-03-25 | Phase 7 | Real-time orchestrator (`realtime_pipeline.py`) deployed |
| 2026-03-26 | Phase 8 | V3 Architecture: Causal LSTM + Temporal Attention logic implementation |
| 2026-03-26 | Phase 8.1 | V3.1 Victory: Aligned with External Model (MAE 256m vs 301m) |

---

## Technical Insights & "Tiniest Details" (Phase 8.1)

### 1. The "Alphabetical Ordering" discovery
During benchmarking, it was discovered that the external model expected the 50 targets (10 zones × 5 horizons) to be sorted **alphabetically by zone name**. Using chronological or zone-index sorting resulted in an "artificial" MAE spike because the model was predicting the wrong runway's visibility.  
**Resolution**: Updated `train_v3.py` and `benchmark_v3.py` to enforce `sorted(list(target_cols))` consistently.

### 2. Temporal Starvation & Attention
Previous models used only the last hidden state of a BiLSTM. However, fog is a "slow-onset, fast-clearing" process.  
**V3.1 Insight**: By switching to a **Temporal Attention Layer** (`TemporalAttention` class), the model can "look back" through its 36-step memory and pick out the exact 10-minute window where Dew Point Depression reached zero, regardless of how many hours ago it happened.

### 3. Architecture Specifics (V3.1)
- **Input Dimensions**: 104 fused features.
- **Lookback window**: 36 steps (6 hours).
- **Hidden Layers**: 3-layer Unidirectional Residual LSTM (384 units per layer).
- **Dropout**: 0.3 (Standard) / 0.15 (Final head).
- **Projection Head**: Linear(384, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 50).
- **Parameters**: 3,573,811 (Optimized for GPU compute).

### 4. Training Hyperparameters
- **Loss**: L1Loss (Direct MAE optimization).
- **Optimizer**: Adam (Weight Decay: 1e-5).
- **Scheduler**: ReduceLROnPlateau (Factor: 0.5, Patience: 10).
- **Batch Size**: 128 (Improved generalization over 512).
- **Hardware**: Accelerated via FP16 Mixed Precision on RTX 5070 Ti.

---

## Final Performance Comparison (Dec 24 - Feb 25)

| Metric | V3.1 (Our Champion) | External (Benchmark) |
|:---|:---:|:---:|
| **MAE (Meters)** | **256.21m** | 300.98m |
| **RMSE (Meters)** | **516.05m** | 563.88m |
| **R² Score** | **0.6249** | 0.5521 |
| **Acc @ 200m** | **81.40% (2024) / 85.70% (2025)** | 65.45% (Dec-Feb Only) |

### Detailed Precision Thresholds (V3.1 Overall)
| Threshold | 2024 (Validation) | 2025 (Test) |
|:---|:---:|:---:|
| **Acc @ 100m** | 76.65% | 80.67% |
| **Acc @ 150m** | 79.32% | 83.52% |
| **Acc @ 200m** | **81.40%** | **85.70%** |
| **Acc @ 250m** | 83.21% | 87.48% |
| **Acc @ 300m** | 84.81% | 88.89% |

---

## Operational Constants
- **Fog Threshold**: 600m (CAT-II Visibility Transition).
- **Operational Horizon**: 6 hours (Standard ATC handover window).
- **Sensor Locations**: 10 canonical zones extracted from KMZ static metadata.

*End of Project Journal*
