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

## Final Project Status
- **Baseline Accomplished**: 301m MAE reduced to 127m.
- **Champion Identified**: V3.1 Residual Attention LSTM.
- **Precision Threshold**: 80.7% Accuracy within 100 meters.
- **Risk Profile**: High precision (83%), low recall (27%) at 600m threshold.

*End of Project Journal*
