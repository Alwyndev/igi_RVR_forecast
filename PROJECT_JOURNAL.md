# PROJECT JOURNAL — IGIA RVR BiLSTM Predictive Model

> **Project**: 6-Hour Ahead RVR Prediction for Indira Gandhi International Airport  
> **Model**: Unified Residual Attention LSTM (V3.1)  
> **Author**: Alwyn  
> **Started**: 2026-03-24  

---

## Technical Performance: V3.1 vs V4 (CNN-LSTM Hybrid)

| Metric          | V3.1 (Attention LSTM) | V4 (CNN-LSTM Hybrid) | Winner |
| :---            | :---:                | :---:                | :---:  |
| **MAE (Meters)**| **127.51m**          | 146.53m              | **V3.1** |
| **RMSE (Meters)**| **370.66m**          | 418.36m              | **V3.1** |
| **R² Score**    | **0.48**             | 0.27                 | **V3.1** |
| **Acc @ 100m**  | **80.73%**           | 80.25%               | **V3.1** |
| **Acc @ 200m**  | **85.75%**           | 82.86%               | **V3.1** |
| **Parameters**  | **3.57M**            | 3.83M (+7.2%)        | **V3.1** |

### Why V3.1 Won
The V4 hybrid model attempted to use 1D-CNN layers for local temporal encoding. However, the benchmark results show that this additional complexity actually introduced smoothing that masked fine-grained temporal gradients. The pure Temporal Attention in V3.1 proved superior at capturing the "fog-trigger" moments in the humidity and temperature sequence.

---

## Project Change Log

| Date | File / Action | Description |
|:---|:---|:---|
| 2026-03-24 | Project Init | Created directory structure, requirements.txt, .gitignore |
| 2026-03-25 | Phase 6 | True Multi-Horizon forecasting (+10m to +6h) developed |
| 2026-03-26 | Phase 8.1 | V3.1 Victory: Aligned with External Model (MAE 256m vs 301m) |
| 2026-03-27 | Phase 11 | External V2 Challenge: V3.1 winner by 1000m+ margin |
| 2026-03-27 | Phase 12 | CNN-LSTM Hybrid Experiment: V3.1 maintains dominance |

---

## Operational Constants
- **Fog Threshold**: 600m (CAT-II Visibility Transition).
- **Operational Horizon**: 6 hours.
- **Sensor Locations**: 10 canonical zones from RWY 09 TDZ to RWY 29 END.

*End of Project Journal*
