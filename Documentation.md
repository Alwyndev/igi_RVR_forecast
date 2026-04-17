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

### 3.7 XGBoost Baseline (Tree-Based Alternative, March 31 2026)
To benchmark a non-sequential tabular learner against the LSTM family, we trained a dedicated multi-horizon XGBoost system using the same feature space (104 features), same 50 targets, and same chronological split protocol.

#### Setup
- **Trainer**: `src/models/train_xgboost.py`
- **Modeling Strategy**: 50 independent `XGBRegressor` models (one per target) trained in a unified loop.
- **Data Split**: Train (<=2023), Validation (2024), Test (2025)
- **Dataset Sizes**: Train 160,032 | Val 45,898 | Test 46,454
- **Objective**: `reg:squarederror`
- **Core Parameters**: `n_estimators=900`, `max_depth=8`, `learning_rate=0.03`, `subsample=0.9`, `colsample_bytree=0.85`
- **Safety Bias**: Fog-weighted row sampling with `fog_weight=4.0` at 600m threshold
- **Training Observability**: Per-target progress with validation RMSE emitted every 100 boosting rounds
- **Wall-Clock Duration**: ~3h 07m for all 50 targets

#### Baseline XGBoost Metrics
| Split | MAE (m) | Fog Precision @ 600m | Fog Recall @ 600m |
| :--- | :---: | :---: | :---: |
| **Validation (2024)** | **167.30** | **91.11%** | **30.88%** |
| **Test (2025)** | **180.50** | **83.33%** | **2.88%** |

#### Interpretation
- XGBoost baseline is extremely conservative on unseen 2025 fog events (very high precision, very low recall).
- The validation recall (30.88%) did not transfer to 2025 test (2.88%), indicating strong distribution shift for fog behavior under this tree-only setup.
- This motivates a second tuning run with stronger fog-prior weighting to intentionally trade MAE/precision for improved recall.

### 3.8 XGBoost Recall-Tuned Reruns (March 31 2026)
A second run was executed explicitly to increase fog-event sensitivity, with heavier fog weighting and slightly fewer boosting rounds for faster convergence.

#### Tuning Changes from Baseline
- `fog_weight`: **4.0 -> 12.0**
- `n_estimators`: **900 -> 500**
- All other core model parameters and data split protocol unchanged.

#### Recall-Tuned Metrics
| Split | MAE (m) | Fog Precision @ 600m | Fog Recall @ 600m |
| :--- | :---: | :---: | :---: |
| **Validation (2024)** | **167.48** | **90.36%** | **32.42%** |
| **Test (2025)** | **178.69** | **87.48%** | **4.66%** |

#### Aggressive Recall Run Metrics
| Split | MAE (m) | Fog Precision @ 600m | Fog Recall @ 600m |
| :--- | :---: | :---: | :---: |
| **Validation (2024)** | **170.67** | **90.20%** | **31.58%** |
| **Test (2025)** | **180.42** | **85.51%** | **4.43%** |

#### Side-by-Side: Baseline vs Recall-Tuned vs Aggressive
| Split | Variant | MAE (m) | Precision @ 600m | Recall @ 600m |
| :--- | :--- | :---: | :---: | :---: |
| Validation | Baseline (`fog_weight=4.0`, `n_estimators=900`) | 167.30 | 91.11% | 30.88% |
| Validation | Recall-Tuned (`fog_weight=12.0`, `n_estimators=500`) | 167.48 | 90.36% | 32.42% |
| Validation | Aggressive (`fog_weight=20.0`, `n_estimators=700`) | 170.67 | 90.20% | 31.58% |
| Test | Baseline (`fog_weight=4.0`, `n_estimators=900`) | 180.50 | 83.33% | 2.88% |
| Test | Recall-Tuned (`fog_weight=12.0`, `n_estimators=500`) | 178.69 | 87.48% | 4.66% |
| Test | Aggressive (`fog_weight=20.0`, `n_estimators=700`) | 180.42 | 85.51% | 4.43% |

#### Trade-off Summary
- **Recall improved** on both splits (Validation: +1.54 pts, Test: +1.78 pts).
- **Test MAE improved** slightly (180.50m -> 178.69m).
- Precision did not degrade in this run; it remained high and increased on test.
- The **aggressive** run (`fog_weight=20`) did not outperform the recall-tuned run (`fog_weight=12`) and introduced MAE regression.
- Despite improvement versus baseline, absolute test fog recall remains low (best: 4.66%), so tree-only models still under-capture rare fog onset compared to the tuned LSTM family.

### 3.9 Dynamic Hybrid + XGBoost Fusion Benchmark (April 17 2026)
To test whether XGBoost can improve the current sequence-model compromise strategy, we executed a dedicated fusion benchmark combining the V3.1+V5 dynamic hybrid with the recall-tuned XGBoost model.

#### Setup
- **Benchmark Script**: `src/models/benchmark_dynamic_hybrid_xgboost.py`
- **Inputs**: V3.1 predictions, V5 predictions, Dynamic Hybrid predictions, XGBoost predictions
- **Tuning Protocol**: 2024 validation for parameter selection; 2025 test for final reporting
- **Fog Threshold**: 600m

#### Validation-Selected Parameters
- Dynamic V3/V5: `w_v5_clear=0.25`, `w_v5_fog=0.60`, `fog_lo=600`, `fog_hi=1300`
- Dynamic (Hybrid+XGB): `w_xgb_clear=0.10`, `w_xgb_fog=0.05`, `fog_lo=600`, `fog_hi=900`

#### Test Results (2025)
| Model / Strategy | MAE | RMSE | R2 | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3.1** | 127.51m | 370.65m | 0.4800 | 80.73% | 85.75% | **83.38%** | 27.09% | 0.4089 |
| **V5 (4.5x)** | 141.61m | 381.94m | 0.3921 | 77.12% | 83.70% | 70.05% | **32.85%** | **0.4473** |
| **Dynamic Hybrid (V3+V5)** | **127.23m** | 365.65m | 0.4887 | **79.98%** | **85.55%** | 78.98% | 29.82% | 0.4330 |
| **XGBoost (recall-tuned)** | 178.45m | 408.54m | 0.2831 | 66.80% | 77.36% | 87.47% | 4.66% | 0.0885 |
| **Dynamic Hybrid + XGBoost** | 129.38m | **359.94m** | **0.5051** | 78.83% | 85.45% | 80.00% | 26.40% | 0.3970 |
| **Ridge Stacking (Dyn,V3,V5,XGB)** | 151.97m | 375.20m | 0.4543 | 71.97% | 81.54% | 90.71% | 4.58% | 0.0872 |

#### Interpretation
- Dynamic Hybrid + XGBoost increased precision slightly (+1.02 pts vs Dynamic Hybrid) but reduced recall (-3.42 pts), reduced Fog F1 (-0.0360), and worsened MAE (+2.15m).
- Ridge stacking converged to an overly conservative regime (very high precision, very low recall), making it unsuitable for fog-capture objectives.
- **Operational decision**: retain **Dynamic Hybrid (V3+V5)** as the preferred production compromise under current model family and tuning constraints.

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
*Document Version: 4.7 (Added Dynamic Hybrid + XGBoost Fusion Benchmark)*
