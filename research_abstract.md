# Research Abstract: Multi-Horizon RVR Forecasting at IGIA

**Title**: Enhancing 6-Hour Ahead Runway Visual Range (RVR) Forecasting at Indira Gandhi International Airport using Temporal Attention Residual LSTMs

**Abstract**:
Accurate, multi-horizon Runway Visual Range (RVR) forecasting is critical for flight safety and operational efficiency at high-traffic hubs like Indira Gandhi International Airport (IGIA), where winter fog frequently disrupts landing categories. This research presents a novel architecture, the **Residual Attention LSTM (V3.1)**, designed to simultaneously predict RVR across 10 canonical runway zones for five distinct horizons (+10m to +6h). 

To address the limitations of standard sequential models, we implemented a 104-feature data fusion pipeline, integrating RVR, METAR (ASOS), Air Quality Index (AQI), and spatial metadata via Haversine-weighted interpolation. Our proposed V3.1 model incorporates a **Temporal Attention mechanism** and **Causal Residual Blocks**, enabling the network to dynamically weight critical fog-onset timesteps within a 36-step (6-hour) lookback window while maintaining strict temporal causality. 

Comparative benchmarking against baseline models developed by a three-member research team shows that the V3.1 model significantly outperforms existing Residual LSTM benchmarks. Specifically, V3.1 achieved a **Mean Absolute Error (MAE) of 256.21m** and an **Accuracy@200m of 85.70%** on an unseen 2024-2025 winter test set, reducing error by approximately 15% compared to the team's primary benchmark (300.98m MAE). Furthermore, experimental analysis of seasonal training (October-March) versus full-year distribution suggests that full-year data exposure provides superior generalization by capturing baseline atmospheric physics during clear-sky periods. These results demonstrate that the integration of temporal attention and multi-source feature fusion provides a state-of-the-art solution for high-precision airport visibility forecasting.

**Keywords**: RVR Forecasting, Temporal Attention, Residual LSTM, IGIA, Multi-Horizon Prediction, Aviation Safety.
