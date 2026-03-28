# IGIA RVR Predictive Modeling — Master Documentation
*A High-Precision 6-Hour Forecasting System for Indira Gandhi International Airport*

---

## 1. Executive Summary
This project delivers a multi-zone, multi-horizon RVR (Runway Visual Range) forecasting system. By integrating 5 years of heterogeneous meteorological data (RVR, METAR, AQI), we developed the **Residual Attention LSTM (V3.1)**, which achieved a state-of-the-art **256m MAE** target, decisively outperforming external benchmarks (301m MAE).

---

## 2. Technical Architecture: Phase 8.1 (Champion Model)

### 2.1 Why Attention + LSTM?
While standard BiLSTMs are effective for short sequences, they suffer from information loss over long 6-hour lookbacks. V3.1 utilizes:
- **Unidirectional Residual Blocks**: Ensures strict temporal causality, preventing "future leakage."
- **Temporal Attention**: A learnable weighting mechanism that enables the model to focus on the exact 10-minute window where Dew Point Depression reached zero (fog formation).
- **Residual Connections**: Stabilizes gradient flow through 3 deep LSTM layers.

### 2.2 Model Specifications
- **Architecture Type**: Residual Attention LSTM
- **Layers**: 3 Hidden Layers (384 units each)
- **Input Features**: 104 (Physics-based, Pollutants, Time, Spatial Lags)
- **Output Head**: Multi-Horizon 50-Neuron (10 Zones × 5 Horizons)
- **Horizons**: +10m, +30m, +1h, +3h, +6h
- **Parameters**: 3.57 Million

---

## 3. Performance & Verification

### 3.1 Hard Winter Window (Dec 2024 - Feb 2025)
This window is historically the most challenging due to persistent dense fog.

| Metric | V3.1 (Attention LSTM) | External Residual LSTM | status |
| :--- | :---: | :---: | :--- |
| **MAE (Meters)** | **256.21m** | 300.98m | ✅ **SOTA WINNER** |
| **RMSE (Meters)** | **516.05m** | 563.88m | ✅ **Superior** |
| **R² Score** | **0.6249** | 0.5521 | ✅ **Superior** |
| **Acc @ 200m** | **81.40% (2024) / 85.70% (2025)** | 65.45% (Dec-Feb Only) | ✅ **Superior** |

### 3.2 Detailed Precision Thresholds (V3.1 Overall)
| Threshold | 2024 (Validation) | 2025 (Test) |
|:---|:---:|:---:|
| **Acc @ 100m** | 76.65% | 80.67% |
| **Acc @ 150m** | 79.32% | 83.52% |
| **Acc @ 200m** | **81.40%** | **85.70%** |
| **Acc @ 250m** | 83.21% | 87.48% |
| **Acc @ 300m** | 84.81% | 88.89% |

### 3.3 Visual Analysis
V3.1 demonstrates a significantly superior ability to track rapid RVR drops (fog onset) compared to the external model.
Refer to: `logs/benchmark_v3_results.png`

### 3.4 Safety-First Variant (V5 Tuned)
To cater to safety-critical operations, a V5 variant was developed utilizing `RVRAsymmetricLoss`. This custom loss mechanism penalizes dangerous over-predictions (predicting clear conditions when actual RVR drops below 600m).
- **V3.1 (Standard)**: 127.51m MAE, 27.09% Fog Recall. Recommended for general, high-reliability operations.
- **V5 (Tuned 4.5x Penalty)**: 141.61m MAE, 32.85% Fog Recall. Recommended for safety-first periods. It achieves a Pareto-optimal sweet spot, catching more unexpected fog onset without drastically degrading overall regression accuracy.

### 3.5 Hybrid Experiment (V3.1 + V5)
#### Hypothesis
Combining V3.1 and V5 should deliver a better balance than either standalone model by preserving low MAE while increasing fog-event sensitivity.

#### Tested Strategies
- **Static Hybrid**: Single fixed blend tuned on 2024 validation (`65% V3.1 + 35% V5`).
- **Dynamic Hybrid**: Fog-risk-aware blend tuned on 2024 validation (`w_v5_clear=0.25`, `w_v5_fog=0.60`, transition band `600m -> 1300m`).

#### Findings on 2025 Test Split
| Model / Strategy | MAE | RMSE | R2 | Fog Precision | Fog Recall | Fog F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3.1** | 127.51m | 370.65m | 0.4800 | **83.38%** | 27.09% | 0.4089 |
| **V5 (4.5x)** | 141.61m | 381.94m | 0.3921 | 70.05% | **32.85%** | **0.4473** |
| **Static Hybrid** | 127.90m | **364.93m** | **0.4891** | 81.90% | 28.03% | 0.4176 |
| **Dynamic Hybrid** | **127.23m** | 365.65m | 0.4887 | 78.98% | 29.82% | 0.4330 |

#### Conclusion
The hypothesis is partially confirmed. Hybridization does not dominate every metric, but the dynamic hybrid provides the best practical compromise: slightly better MAE than V3.1 and higher recall than V3.1, with acceptable precision trade-off.

### 3.6 Targeted Subset Evaluation (2 TDZ + 2 MID)
To validate runway-critical behavior at representative touchdown and midpoint sensors, we ran a focused benchmark on:
- TDZ: `09_TDZ`, `11_TDZ`
- MID: `MID_2810`, `MID_2911`

#### Findings on 2025 Test Split (Subset Only)
| Model / Strategy | MAE | RMSE | R2 | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3.1** | 130.83m | 339.86m | 0.6476 | **77.08%** | **83.23%** | **75.39%** | 17.11% | 0.2789 |
| **V5 (4.5x)** | 146.09m | 355.04m | 0.6121 | 73.60% | 81.01% | 40.75% | **23.42%** | 0.2975 |
| **Dynamic Hybrid** | **129.89m** | **333.89m** | **0.6585** | 76.52% | 83.11% | 56.04% | 20.70% | **0.3023** |

#### Subset Interpretation
- Dynamic Hybrid is best on global regression quality for this subset (MAE, RMSE, R2) and best on Fog F1.
- V3.1 remains strongest for precision-oriented operation on these zones (best Fog Precision and slight edge in Acc@100m/200m).
- V5 remains the highest-recall option, but with a larger false-alarm penalty on this subset.

---

## 4. Key Discovery: The Alpha-Ordering Standard
A critical project detail: Data alignment. The 2024-25 RVR dataset requires targets to be sorted **alphabetically by zone** (`09_TDZ, 10_NEW, 11_BEG...`). All internal scripts (`train_v3.py`, `benchmark_v3.py`) have been synchronized to this standard to ensure fair evaluation of the 50-neuron output.

---

## 5. Operations & Scaling

### 5.1 Training Workflow
```powershell
# Run the V3.1 training with GPU acceleration and Plateau scheduler
python src.models.train_v3.py --no-wandb
```

### 5.2 Real-Time Integration
The champion model is integrated into `realtime_pipeline.py`. It polls the `Latest Data/` folder every 10 minutes and generates the interactive `igia_rvr_dashboard_multi.html`.

---
*Document Version: 4.3 (Subset Benchmark Update)*
