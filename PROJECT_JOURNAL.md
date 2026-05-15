# PROJECT JOURNAL — IGIA RVR BiLSTM Predictive Model

> **Project**: 6-Hour Ahead RVR Prediction for Indira Gandhi International Airport  
> **Model**: Unified Residual Attention LSTM (V3.1)  
> **High-Recall Variant**: V5 (Asymmetric Loss)
> **Author**: Alwyn  
> **Started**: 2026-03-24  

---

## Safety Performance Comparison (600m Threshold)

| Model             | Global MAE | Fog Recall (Sensitivity) | Fog Precision (Reliability) | F1-Score | Status |
| :---              | :---:      | :---:                    | :---:                       | :---:    | :--- |
| **V3.1 (Standard)**| **127.51m**| 27.09%                   | **83.38%**                  | 0.4089   | Champion (MAE) |
| **V5 (Old - 8.0x)**| 159.74m    | **41.53%**               | 65.07%                      | **0.5070**| High Recall |
| **V5 (Tuned - 4.5x)**| 141.61m  | 32.85%                   | 70.05%                      | 0.4473   | Balanced Safety |

### V5 Technical Insight & Tuning
Initially, we implemented a `RVRAsymmetricLoss` with an aggressive 8.0x penalty for over-predicting visibility during fog windows. This successfully shifted the model's bias towards safety (41.5% Recall) but heavily degraded our global MAE to ~160m. 
By tuning the asymmetry penalty down to **4.5x**, we established a Pareto-optimal sweet spot: clawing back ~18 meters of MAE (bringing it down to 141.6m) while maintaining a better fog recall (32.85%) and better F1 (0.4473) than the standard V3.1 model. This provides a balanced alternative to the V3.1 baseline.

### Hybrid Ensemble Benchmark (V3.1 + V5)
We evaluated whether ensembling the high-accuracy V3.1 with the safety-focused V5 could improve overall operational performance on the 2025 test split.

#### Hypothesis
A hybrid of V3.1 and V5 should outperform both individual models by combining V3.1's low-error regression behavior with V5's stronger fog sensitivity.

| Model / Strategy | MAE | RMSE | R2 | Fog Precision | Fog Recall | Fog F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3.1** | 127.51m | 370.65m | 0.4800 | **83.38%** | 27.09% | 0.4089 |
| **V5 (4.5x)** | 141.61m | 381.94m | 0.3921 | 70.05% | **32.85%** | **0.4473** |
| **Static Hybrid (65% V3 + 35% V5)** | 127.90m | **364.93m** | **0.4891** | 81.90% | 28.03% | 0.4176 |
| **Dynamic Hybrid (risk-aware)** | **127.23m** | 365.65m | 0.4887 | 78.98% | 29.82% | 0.4330 |

### Dynamic Hybrid Configuration
Dynamic blending was tuned on the 2024 validation period using a fog-risk-aware weight schedule:
- `w_v5_clear = 0.25`
- `w_v5_fog = 0.60`
- `fog_lo = 600m`, `fog_hi = 1300m`

This means V5 contributes only lightly during clear conditions but gains influence as predicted visibility moves into fog-like bands. Net effect: better global MAE than V3.1 and better fog recall than V3.1, while preserving much of V3.1 precision.

#### Findings
- The static hybrid improved RMSE and R2 versus both individual models, but did not beat V3.1 on MAE.
- The dynamic hybrid achieved the best MAE (127.23m), slightly better than V3.1 (127.51m).
- The dynamic hybrid increased fog recall versus V3.1 (29.82% vs 27.09%), though still below V5 (32.85%).
- Conclusion: the hypothesis is partially validated. Hybridization improves trade-off quality, and dynamic weighting is the strongest compromise strategy.

### Focused Benchmark: 2 TDZ + 2 MID
To stress-test operationally important runway points, we evaluated only:
- `09_TDZ`, `11_TDZ`
- `MID_2810`, `MID_2911`

| Model / Strategy | MAE | RMSE | R2 | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3.1** | 130.83m | 339.86m | 0.6476 | **77.08%** | **83.23%** | **75.39%** | 17.11% | 0.2789 |
| **V5 (4.5x)** | 146.09m | 355.04m | 0.6121 | 73.60% | 81.01% | 40.75% | **23.42%** | 0.2975 |
| **Dynamic Hybrid** | **129.89m** | **333.89m** | **0.6585** | 76.52% | 83.11% | 56.04% | 20.70% | **0.3023** |

#### Subset Conclusions
- Dynamic Hybrid leads overall for this subset by achieving the best MAE/RMSE/R2 and best Fog F1.
- V3.1 is still preferred when false alarms must be minimized (highest precision and top threshold accuracies).
- V5 is still preferred when the sole priority is maximum fog capture (highest recall).

---

## Technical Performance: V3.1 vs V4

| Metric          | V3.1 (Attention LSTM) | V4 (CNN-LSTM Hybrid) | Winner |
| :---            | :---:                | :---:                | :---:  |
| **MAE (Meters)**| **127.51m**          | 146.53m              | **V3.1** |
| **RMSE (Meters)**| **370.66m**          | 418.36m              | **V3.1** |
| **Acc @ 100m**  | **80.73%**           | 80.25%               | **V3.1** |
| **Acc @ 200m**  | **85.75%**           | 82.86%               | **V3.1** |

---

## XGBoost Baseline Run (March 31, 2026)

To test whether a pure tabular gradient-boosting family can complement or replace sequence models, we trained a full 50-target XGBoost benchmark.

### Training Configuration
- **Script**: `src/models/train_xgboost.py`
- **Approach**: One XGBoost regressor per target (50 total)
- **Features / Targets**: 104 / 50
- **Split**: Train <=2023, Val=2024, Test=2025
- **Rows**: Train 160,032 | Val 45,898 | Test 46,454
- **Params**: `n_estimators=900`, `max_depth=8`, `learning_rate=0.03`, `subsample=0.9`, `colsample_bytree=0.85`
- **Fog Weighting**: `fog_weight=4.0` (rows with any target <600m)
- **Runtime**: ~3h 07m

### Exact Metrics (Baseline XGBoost)
| Split | MAE | Fog Precision | Fog Recall |
| :--- | :---: | :---: | :---: |
| **Validation (2024)** | 167.30m | 91.11% | 30.88% |
| **Test (2025)** | 180.50m | 83.33% | 2.88% |

### Diagnostic Notes
- Validation looked promising on recall, but generalization on 2025 fog events collapsed (2.88% recall).
- Failure mode is over-conservative prediction under test-shift: strong precision, near-zero sensitivity.
- Next action: run a recall-biased tuning pass (higher fog weighting) and compare side-by-side.

## XGBoost Recall-Tuned Runs (March 31, 2026)

To increase fog capture, we executed a second full run with stronger fog-prior weighting.

### Tuning Deltas
- `fog_weight`: 4.0 -> 12.0
- `n_estimators`: 900 -> 500
- Data split, features, and core tree hyperparameters unchanged.

### Exact Metrics (Recall-Tuned)
| Split | MAE | Fog Precision | Fog Recall |
| :--- | :---: | :---: | :---: |
| **Validation (2024)** | 167.48m | 90.36% | 32.42% |
| **Test (2025)** | 178.69m | 87.48% | 4.66% |

### Exact Metrics (Aggressive Recall)
| Split | MAE | Fog Precision | Fog Recall |
| :--- | :---: | :---: | :---: |
| **Validation (2024)** | 170.67m | 90.20% | 31.58% |
| **Test (2025)** | 180.42m | 85.51% | 4.43% |

### Side-by-Side (Baseline vs Recall-Tuned vs Aggressive)
| Split | Variant | MAE | Fog Precision | Fog Recall |
| :--- | :--- | :---: | :---: | :---: |
| Validation | Baseline (4.0, 900 trees) | 167.30m | 91.11% | 30.88% |
| Validation | Recall-Tuned (12.0, 500 trees) | 167.48m | 90.36% | 32.42% |
| Validation | Aggressive (20.0, 700 trees) | 170.67m | 90.20% | 31.58% |
| Test | Baseline (4.0, 900 trees) | 180.50m | 83.33% | 2.88% |
| Test | Recall-Tuned (12.0, 500 trees) | 178.69m | 87.48% | 4.66% |
| Test | Aggressive (20.0, 700 trees) | 180.42m | 85.51% | 4.43% |

### Takeaways
- Recall improved on both validation and test, which was the tuning objective.
- Absolute test recall is still very low for operations where missed fog events are costly.
- In this configuration, MAE also improved slightly on test; this run did not force the expected MAE penalty.
- The aggressive pass did not beat the recall-tuned pass; it reduced recall slightly and worsened MAE.
- XGBoost remains a useful tabular benchmark, but sequence-aware LSTM variants still dominate safety-relevant fog sensitivity in this project.

## Dynamic Hybrid + XGBoost Fusion Experiment (April 17, 2026)

To test whether tree-based predictions can improve the existing V3.1+V5 dynamic hybrid, we ran a dedicated fusion benchmark:

- **Script**: `src/models/benchmark_dynamic_hybrid_xgboost.py`
- **Compared Models**: V3.1, V5, Dynamic Hybrid (V3+V5), XGBoost (recall-tuned), Dynamic Hybrid + XGBoost (risk-aware), Ridge Stacking (Dyn,V3,V5,XGB)
- **Split**: Validation=2024 (tuning), Test=2025 (final reporting)

### Validation-selected Parameters
- Dynamic V3/V5: `w_v5_clear=0.25`, `w_v5_fog=0.60`, `fog_lo=600m`, `fog_hi=1300m`
- Dynamic (Hybrid+XGB): `w_xgb_clear=0.10`, `w_xgb_fog=0.05`, `fog_lo=600m`, `fog_hi=900m`

### Test Metrics (2025)
| Model / Strategy | MAE | RMSE | R2 | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3.1** | 127.51m | 370.65m | 0.4800 | 80.73% | 85.75% | **83.38%** | 27.09% | 0.4089 |
| **V5 (4.5x)** | 141.61m | 381.94m | 0.3921 | 77.12% | 83.70% | 70.05% | **32.85%** | **0.4473** |
| **Dynamic Hybrid (V3+V5)** | **127.23m** | 365.65m | 0.4887 | **79.98%** | **85.55%** | 78.98% | 29.82% | 0.4330 |
| **XGBoost (recall-tuned)** | 178.45m | 408.54m | 0.2831 | 66.80% | 77.36% | 87.47% | 4.66% | 0.0885 |
| **Dynamic Hybrid + XGBoost** | 129.38m | **359.94m** | **0.5051** | 78.83% | 85.45% | 80.00% | 26.40% | 0.3970 |
| **Ridge Stack (Dyn,V3,V5,XGB)** | 151.97m | 375.20m | 0.4543 | 71.97% | 81.54% | 90.71% | 4.58% | 0.0872 |

### Outcome
- Dynamic Hybrid + XGBoost improved precision slightly (**+1.02 pts**) but reduced recall (**-3.42 pts**) and worsened MAE (**+2.15m**) versus Dynamic Hybrid.
- Ridge stacking collapsed to a high-precision / near-zero-recall regime and is not operationally suitable for fog-capture goals.
- **Decision**: keep **Dynamic Hybrid (V3+V5)** as the deployment-favored compromise for now.

---

## Final Project Status
- **Baseline Accomplished**: 301m MAE reduced to 127m.
- **Champion Identified**: V3.1 Residual Attention LSTM.
- **Precision Threshold**: 80.7% Accuracy within 100 meters.
- **Risk Profile**: High precision (83%), low recall (27%) at 600m threshold.
- **Latest Ensemble Finding**: Adding XGBoost to Dynamic Hybrid did not improve the primary safety trade-off (recall/F1).

---

## Real-Time Production Deployment (May 15, 2026)

To operationalize the Dynamic Hybrid model for live predictions, we developed a fully autonomous data ingestion and inference pipeline.

### 1. Data Scraper (`scrape_realtime.py`)
- **WiFEX Integration**: Pulls ambient visibility and meteorological data (temp, rh, wind) directly from the IITM WiFEX AWS endpoints. Handles missing sensor logic (e.g., falling back from 10m to 20m anemometers).
- **Playwright DOM Scraping**: The new IMD RVR portal (`live-rvr`) serves dynamic data via STOMP WebSockets, which is inaccessible to standard HTTP requests. We integrated Playwright to spin up a headless Chromium browser, navigate the DOM, select "New Delhi Airport", and extract the live RVR values across all 10 zones.

### 2. Feature Engineering Pipeline (`preprocess_realtime.py`)
- Resamples the raw 12-hour buffer to a strict 10-minute frequency grid.
- Derives complex physical metrics (e.g., Magnus formula for Dewpoint Depression) and rolling statistics (1h, 3h, 6h lags & std dev).
- **Graceful Degradation**: If WiFEX or the RVR portal goes offline, the pipeline automatically fills NaNs with the training-set means (extracted from `scaler_X.pkl`), ensuring the model always receives exactly 104 valid features and never crashes the API.

### 3. Application Integration (`app.py`)
- Added `APScheduler` to the primary Flask application.
- A background job autonomously executes the scrape-and-preprocess pipeline every 10 minutes, entirely decoupled from the API request thread.
- The `MultiHorizonEngine` consumes the resulting `model_input.parquet` to rebuild the interactive Folium dashboard (`/map`) and update the live JSON API (`/predictions_multi`) consumed by the Flutter application. 
- The system is now 100% automated and ready for live ATC testing.

*End of Project Journal*
