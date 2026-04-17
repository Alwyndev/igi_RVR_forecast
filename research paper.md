# Enhancing Multi-Horizon Runway Visual Range Forecasting at IGIA with Residual Attention LSTM and Safety-Aware Hybridization

## Abstract

Reliable 6-hour Runway Visual Range (RVR) forecasting is a core operational requirement for high-traffic airports affected by recurrent low-visibility episodes. This paper presents an end-to-end forecasting framework for Indira Gandhi International Airport (IGIA), Delhi, that predicts visibility for 10 runway zones across five lead times (+10m, +30m, +1h, +3h, +6h). We develop and evaluate a Residual Attention LSTM architecture (V3.1) trained on multi-source, multi-year meteorological and environmental data. The pipeline integrates 104 engineered features derived from RVR observations, METAR/ASOS weather fields, air quality indicators, temporal encodings, and spatial interpolation using Haversine-weighted station fusion.

Across challenging winter windows, V3.1 achieves 256.21 m MAE and 85.70% Accuracy at 200 m on unseen 2024-2025 conditions, improving substantially over an external Residual LSTM benchmark (300.98 m MAE). In full-year aggregate testing, V3.1 reaches 127.51 m MAE with strong precision under fog-threshold classification, though with moderate fog-event recall. To address safety sensitivity, we introduce a high-recall V5 variant using asymmetric loss and evaluate static and dynamic hybrid ensembles (V3.1 + V5). The dynamic risk-aware blend improves the trade-off frontier, achieving 127.23 m MAE while increasing fog recall relative to V3.1.

Results indicate that temporal attention, strict causal residual sequence modeling, and hybrid decision blending provide a practical and high-performance pathway for operational airport visibility forecasting.

## 1. Introduction

Low-visibility operations remain one of the most critical constraints in airport traffic management. At IGIA, dense winter fog frequently causes rapid visibility deterioration, disrupting approach sequencing, runway utilization, and airline schedule reliability. The forecasting objective is therefore not only regression accuracy, but also timely capture of fog onset at operationally meaningful horizons.

Traditional persistence or purely tabular models often underperform in this setting because fog dynamics exhibit temporal lag structure, nonlinear thresholds, and localized spatial propagation. Sequence-aware neural models can better encode these effects, but long lookback windows introduce optimization and attribution challenges.

This work addresses these issues with a unified multi-zone, multi-horizon architecture and presents a complete operational workflow from data fusion to dashboard outputs. The major contributions are:

1. A 50-target forecasting setup for 10 runway zones and 5 horizons in one forward pass.
2. A 104-feature fusion pipeline combining visibility, weather, air-quality, temporal, and spatial signals.
3. A Residual Attention LSTM (V3.1) that preserves temporal causality while emphasizing fog-relevant timesteps.
4. Safety-aware post-model strategy design via asymmetric-loss training (V5) and dynamic hybrid blending.
5. Comprehensive evaluation across seasonal windows, full-year splits, and runway-critical subsets.

## 2. Problem Formulation

Let x_t denote the feature vector at time t (10-minute cadence), and let y_t denote future RVR values for all zone-horizon targets. For each timestamp, the model receives a fixed lookback sequence X_t = {x_(t-L+1), ..., x_t} with L = 36 steps (6 hours), and predicts a 50-dimensional output vector:

Y_hat_t = f_theta(X_t) in R^50

where the 50 outputs correspond to 10 runway zones x 5 lead times. Model training optimizes multi-target regression error under chronologically separated train/validation/test partitions.

In addition to regression metrics (MAE, RMSE, R2), we evaluate event-level safety behavior at a 600 m fog threshold through precision, recall, and F1 to reflect operational risk asymmetry.

## 3. Data and Feature Engineering

### 3.1 Data Sources

The pipeline combines multiple synchronized sources:

1. RVR sensor observations from runway-adjacent zones.
2. METAR/ASOS meteorological reports and derived weather variables.
3. Air quality indicators (AQI-related fields) as fog-proxy and atmospheric context.
4. Spatial metadata for sensor/station geometry.

### 3.2 Feature Space

A total of 104 features are constructed, including:

1. Raw and lagged visibility/weather variables.
2. Temporal cyclic encodings (hour/day/seasonal periodicity).
3. Physics-informed transforms (for example, fog-supportive thermodynamic regimes).
4. Spatially interpolated station signals via Haversine-distance weighting.

The design goal is to expose both short-term turbulence signatures and slower atmospheric drift patterns to sequence models.

### 3.3 Target Ordering and Alignment

A key reproducibility requirement in this project is strict alphabetical zone ordering for the 50-target head. All training and benchmarking scripts enforce a canonical target layout (for example: 09_TDZ, 10_NEW, 11_BEG, ...). This avoids silent evaluation mismatch and is essential for fair model comparisons.

## 4. Model Architecture

### 4.1 Residual Attention LSTM (V3.1)

V3.1 is a unidirectional causal sequence model with three hidden layers (384 units each), residual pathways, and temporal attention over encoder states. The architecture is designed to satisfy three constraints simultaneously:

1. Temporal causality: no future leakage in sequence processing.
2. Deep-trace stability: residual routing mitigates gradient degradation over long lookbacks.
3. Fog-window focus: attention weights highlight critical onset intervals.

The output layer is a dense 50-neuron regression head, jointly producing all zone-horizon forecasts.

### 4.2 Parameterization

Core model characteristics:

1. Input dimensionality: 104 features.
2. Sequence length: 36 timesteps.
3. Hidden width: 384.
4. Depth: 3 recurrent layers with residual structure.
5. Total parameters: approximately 3.57 million.

### 4.3 Safety-Aware Variant (V5)

To reduce dangerous over-predictions in fog, V5 modifies training objective behavior using an asymmetric penalty regime. Overestimation under low-visibility conditions is penalized more strongly than underestimation, shifting the model toward higher fog detection sensitivity. Penalty tuning reveals a clear precision-recall-MAE trade-off.

## 5. Experimental Protocol

### 5.1 Temporal Splits

Experiments follow strict chronological partitioning:

1. Training: historical years up to 2023.
2. Validation: 2024.
3. Test: 2025.

Season-focused analyses additionally isolate hard winter windows (Dec-Feb) to stress dense-fog performance.

### 5.2 Metrics

Regression quality:

1. MAE (meters).
2. RMSE (meters).
3. R2.
4. Accuracy at absolute error thresholds (100 m, 150 m, 200 m, 250 m, 300 m).

Safety classification at 600 m threshold:

1. Fog precision.
2. Fog recall.
3. Fog F1.

### 5.3 Baselines and Comparators

Compared systems include:

1. External Residual LSTM benchmark.
2. CNN-LSTM hybrid (V4).
3. XGBoost multi-target family (independent regressor per target, 50 models).
4. V5 asymmetric-loss sequence variant.
5. Static and dynamic V3.1 + V5 hybrids.

## 6. Results

### 6.1 Winter-Window Benchmarking

On hard winter conditions (Dec 2024 - Feb 2025), V3.1 outperforms the external Residual LSTM baseline:

| Model | MAE (m) | RMSE (m) | R2 | Acc@200m |
| :--- | ---: | ---: | ---: | ---: |
| V3.1 Residual Attention LSTM | 256.21 | 516.05 | 0.6249 | 85.70% (2025 winter) |
| External Residual LSTM | 300.98 | 563.88 | 0.5521 | 65.45% (Dec-Feb only) |

This corresponds to approximately 15% MAE reduction versus the external benchmark.

### 6.2 Overall V3.1 Quality

Across broader evaluation, V3.1 reports:

1. MAE: 127.51 m.
2. RMSE: 370.65 m.
3. R2: 0.4800.
4. Accuracy thresholds up to 85.75% at 200 m (2025-level reporting context).

The model consistently offers strong precision-oriented behavior for operational reliability, especially when false alarms carry high cost.

### 6.3 V3.1 vs V4 (CNN-LSTM)

| Metric | V3.1 | V4 | Winner |
| :--- | ---: | ---: | :--- |
| MAE (m) | 127.51 | 146.53 | V3.1 |
| RMSE (m) | 370.66 | 418.36 | V3.1 |
| Acc@100m | 80.73% | 80.25% | V3.1 |
| Acc@200m | 85.75% | 82.86% | V3.1 |

Attention plus residual causal recurrence outperforms the tested CNN-LSTM hybrid on both absolute error and threshold accuracy.

### 6.4 Safety Trade-off: V3.1 vs V5

At the 600 m fog threshold:

| Model | MAE (m) | Fog Precision | Fog Recall | Fog F1 |
| :--- | ---: | ---: | ---: | ---: |
| V3.1 (standard) | 127.51 | 83.38% | 27.09% | 0.4089 |
| V5 tuned (4.5x asymmetric) | 141.61 | 70.05% | 32.85% | 0.4473 |

Interpretation:

1. V3.1 is stronger for precision and aggregate regression quality.
2. V5 captures more fog events, improving recall and F1.
3. A single model may be suboptimal for all operational regimes.

### 6.5 Hybridization Outcomes

Hybrid blends were evaluated to combine V3.1 accuracy with V5 sensitivity:

| Strategy | MAE (m) | RMSE (m) | R2 | Fog Precision | Fog Recall | Fog F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3.1 | 127.51 | 370.65 | 0.4800 | 83.38% | 27.09% | 0.4089 |
| V5 (4.5x) | 141.61 | 381.94 | 0.3921 | 70.05% | 32.85% | 0.4473 |
| Static Hybrid (65/35) | 127.90 | 364.93 | 0.4891 | 81.90% | 28.03% | 0.4176 |
| Dynamic Hybrid (risk-aware) | 127.23 | 365.65 | 0.4887 | 78.98% | 29.82% | 0.4330 |

The dynamic schedule (light V5 weight in clear regimes, heavier in fog-like regimes) provides the best compromise: best MAE overall and higher recall than V3.1.

### 6.6 Runway-Critical Subset (2 TDZ + 2 MID)

Focused evaluation on 09_TDZ, 11_TDZ, MID_2810, MID_2911:

| Strategy | MAE (m) | RMSE (m) | R2 | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3.1 | 130.83 | 339.86 | 0.6476 | 77.08% | 83.23% | 75.39% | 17.11% | 0.2789 |
| V5 (4.5x) | 146.09 | 355.04 | 0.6121 | 73.60% | 81.01% | 40.75% | 23.42% | 0.2975 |
| Dynamic Hybrid | 129.89 | 333.89 | 0.6585 | 76.52% | 83.11% | 56.04% | 20.70% | 0.3023 |

Dynamic Hybrid leads this subset in MAE, RMSE, R2, and Fog F1.

### 6.7 XGBoost Baseline and Recall Tuning

XGBoost was trained as a tabular alternative (50 independent regressors, same 104-feature input space).

Baseline vs tuned fog weighting results:

| Variant | Test MAE (m) | Test Fog Precision | Test Fog Recall |
| :--- | ---: | ---: | ---: |
| Baseline (fog_weight 4.0) | 180.50 | 83.33% | 2.88% |
| Recall-tuned (fog_weight 12.0) | 178.69 | 87.48% | 4.66% |
| Aggressive (fog_weight 20.0) | 180.42 | 85.51% | 4.43% |

Even with tuning, absolute fog recall remains far below sequence-model variants, indicating that temporal context learning is central for rare fog-onset capture.

### 6.8 Fusion Experiment: Dynamic Hybrid + XGBoost

To test whether tree-based predictions can improve the established dynamic hybrid (V3+V5), we evaluated two additional ensemble strategies on the same 2025 test split:

1. Dynamic Hybrid + XGBoost (risk-aware blending with validation-tuned weights).
2. Ridge stacking over four predictors (Dynamic Hybrid, V3.1, V5, XGBoost).

Validation-selected parameters were:

1. Dynamic V3/V5: w_v5_clear = 0.25, w_v5_fog = 0.60, fog_lo = 600 m, fog_hi = 1300 m.
2. Dynamic (Hybrid+XGB): w_xgb_clear = 0.10, w_xgb_fog = 0.05, fog_lo = 600 m, fog_hi = 900 m.

| Strategy | MAE (m) | RMSE (m) | R2 | Acc@100m | Acc@200m | Fog Precision | Fog Recall | Fog F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic Hybrid (V3+V5) | 127.23 | 365.65 | 0.4887 | 79.98% | 85.55% | 78.98% | 29.82% | 0.4330 |
| Dynamic Hybrid + XGBoost | 129.38 | 359.94 | 0.5051 | 78.83% | 85.45% | 80.00% | 26.40% | 0.3970 |
| Ridge Stacking (Dyn,V3,V5,XGB) | 151.97 | 375.20 | 0.4543 | 71.97% | 81.54% | 90.71% | 4.58% | 0.0872 |

Interpretation:

1. Adding XGBoost to Dynamic Hybrid improved precision slightly (+1.02 points) but degraded recall (-3.42 points), Fog F1 (-0.0360), and MAE (+2.15 m).
2. Ridge stacking collapsed to a highly conservative regime with near-zero fog sensitivity.
3. Under current feature/model constraints, Dynamic Hybrid (V3+V5) remains the best practical precision-recall compromise.

## 7. Discussion

### 7.1 Why V3.1 Works

Performance gains appear to come from three interacting mechanisms:

1. Rich fused feature space captures both immediate and latent atmospheric drivers.
2. Causal residual recurrence preserves long-range temporal dependencies.
3. Attention isolates high-value timesteps near fog transition boundaries.

### 7.2 Reconciling Metric Regimes

The project reports two complementary views:

1. Hard winter-window performance, where absolute errors are naturally larger and model robustness is stress-tested.
2. Full-year aggregate performance, where clear-weather periods improve global MAE but can mask safety asymmetry.

This paper treats both as necessary: winter windows for stress realism and full-year metrics for deployment representativeness.

### 7.3 Precision vs Recall in Operations

Aviation deployment decisions depend on risk appetite:

1. Precision-first mode (V3.1): minimizes false alarms, better for stable throughput.
2. Recall-first mode (V5): captures more low-visibility events, better for conservative safety posture.
3. Hybrid mode (dynamic): strongest practical compromise under mixed operational conditions.

## 8. Operationalization

The trained system is integrated into a real-time pipeline that polls updated inputs on a 10-minute cadence and writes an interactive multi-horizon dashboard for controller-facing situational awareness. This closes the loop between model inference and decision support.

## 9. Limitations

Current limitations include:

1. Domain shift between validation and test-year fog characteristics.
2. Moderate recall ceiling for the precision-optimized champion model.
3. Dependence on upstream data quality and temporal synchronization.
4. Need for richer uncertainty quantification for threshold-triggered actions.

## 10. Future Work

Planned extensions:

1. Probabilistic forecasting with calibrated predictive intervals.
2. Multi-objective training to jointly optimize MAE and fog recall.
3. Regime-adaptive ensembling with learned gating rather than hand-tuned thresholds.
4. Transfer and adaptation studies across additional airports and climatological regimes.
5. Explainability diagnostics for attention trajectories and event attribution.

## 11. Conclusion

This study demonstrates that high-fidelity multi-horizon RVR forecasting at IGIA benefits from a sequence-first design with strict causality, temporal attention, and feature-rich data fusion. The V3.1 Residual Attention LSTM establishes strong accuracy and precision performance, while V5 and hybrid blending improve safety sensitivity where fog-miss cost is high. Additional fusion with XGBoost improves some regression/correlation metrics but degrades the primary fog-capture trade-off (recall/F1), reinforcing Dynamic Hybrid (V3+V5) as the strongest operational compromise in this project stage. The resulting framework remains technically competitive and operationally actionable, and it defines clear next steps for constrained-recall ensemble optimization.

## References (Draft)

[1] ICAO and AAI operational guidance on low-visibility procedures.
[2] Prior literature on LSTM-based meteorological and visibility forecasting.
[3] Research on attention mechanisms in time-series forecasting.
[4] Works on asymmetric loss design for safety-critical regression.
[5] XGBoost documentation and gradient boosting references.

---

### Appendix A: Reproducibility Snapshot (Draft)

1. Forecast cadence: 10 minutes.
2. Lookback window: 36 steps (6 hours).
3. Targets: 10 zones x 5 horizons = 50 outputs.
4. Input features: 104.
5. Core split: Train <=2023, Validation 2024, Test 2025.
6. Primary thresholded safety analysis: 600 m fog boundary.

### Appendix B: Suggested Submission Metadata (Draft)

1. Domain: AI for Aviation Safety and Operations.
2. Paper type: Applied ML system study with deployment integration.
3. Candidate venues: transportation analytics, atmospheric informatics, and AI systems engineering tracks.
