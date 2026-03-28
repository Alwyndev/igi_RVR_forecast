# PROJECT JOURNAL — IGIA RVR BiLSTM Predictive Model

> **Project**: 6-Hour Ahead RVR Prediction for Indira Gandhi International Airport  
> **Model**: Unified Residual Attention LSTM (V3.1)  
> **High-Recall Variant**: V5 (Asymmetric Loss)
> **Author**: Alwyn  
> **Started**: 2026-03-24  

---

## Safety Performance Comparison (600m Threshold)

| Model             | Fog Recall (Sensitivity) | Fog Precision (Reliability) | F1-Score | Status |
| :---              | :---:                    | :---:                       | :---:    | :--- |
| **V3.1 (Standard)**| 27.09%                   | **83.38%**                  | 0.4089   | Champion (MAE) |
| **V5 (High-Recall)**| **41.53%**               | 65.07%                      | **0.5070**| **Safety Specialist** |

### V5 Technical Insight
By implementing a `RVRAsymmetricLoss` with a 8.0x penalty for over-predicting visibility during fog windows, we successfully shifted the model's bias towards safety. While precision decreased (more false alarms), the 1.5x gain in recall significantly reduces the risk of missed CAT-II/III visibility events.

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
