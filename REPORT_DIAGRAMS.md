# Recommended Images and Diagrams for Internship Report

Here is a curated list of diagrams and images that perfectly illustrate the technical depth of your project. 

> **Tip for Visibility:** If your markdown viewer makes these look small, copy the code blocks below and paste them into [Mermaid Live](https://mermaid.live/). From there, you can easily download them as high-resolution PNGs!

## 1. High-Level System Architecture
*This diagram illustrates the end-to-end flow of data from the sensor networks to the Google Cloud backend, and finally to the Flutter clients used by ATC.*

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'arial', 'primaryColor': '#e8f4f8', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    subgraph Data Sources
        S1[IMD RVR Portal]
        S2[WiFEX Weather Station]
    end

    subgraph Google Cloud Run
        API[Flask API & Scheduler]
        SCR[Playwright Scraper]
        PRE[Preprocessor: 104 Features]
        ENG[Dynamic Hybrid Inference Engine]
        
        API -.->|Triggers every 10 min| SCR
        SCR --> PRE
        PRE --> ENG
    end

    subgraph Flutter Client
        APP[Multi-Platform App]
        UI1[Interactive Map]
        UI2[Horizon Slider]
    end

    S1 -->|STOMP WebSockets| SCR
    S2 -->|REST / HTML| SCR
    ENG -->|/predictions_multi| API
    API -->|HTTPS / JSON| APP
    APP --> UI1
    APP --> UI2

    classDef gcp fill:#e8f0fe,stroke:#4285f4,stroke-width:3px,color:#000;
    classDef client fill:#e1f5fe,stroke:#02569b,stroke-width:3px,color:#000;
    classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:3px,color:#000;
    
    class API,SCR,PRE,ENG gcp;
    class APP,UI1,UI2 client;
    class S1,S2 data;
```

---

## 2. Dynamic Hybrid Ensemble Logic (Model Architecture)
*This diagram highlights your major machine learning contribution: combining the high-accuracy V3.1 model with the safety-first V5 model.*

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'arial'}}}%%
flowchart TD
    IN(Normalized 36-timestep Window)
    
    subgraph Models
        V3[V3.1 Residual Attention LSTM]
        V5[V5 Asymmetric Loss LSTM]
    end

    subgraph Dynamic Blending
        RISK{Risk-Aware Weight Schedule}
        CLEAR[Clear Conditions: V5 Weight 0.25]
        FOG[Fog Conditions: V5 Weight 0.60]
    end

    OUT(((Final Multi-Horizon Prediction)))

    IN --> V3
    IN --> V5
    V3 --> RISK
    V5 --> RISK
    
    RISK -->|> 1300m| CLEAR
    RISK -->|< 600m| FOG
    
    CLEAR --> OUT
    FOG --> OUT
    
    style V3 fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    style V5 fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000
    style OUT fill:#e8f5e9,stroke:#2e7d32,stroke-width:4px,color:#000
```

---

## 3. Real-Time Data Pipeline
*This flowchart breaks down the complex data preprocessing steps required to feed the models, perfect for the "Implementation" section of your report.*

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'arial'}}}%%
flowchart LR
    A[(Raw 12-Hour Buffer)] --> B[Resampling]
    B -->|Strict 10-min grid| C{Feature Engineering}
    
    C --> D[Physical Metrics]
    C --> E[Rolling Statistics]
    
    D -.->|Magnus Dewpoint, Wind Chills| F[104-Feature Vector]
    E -.->|1h/3h/6h lags & std dev| F
    
    F -->|Graceful NaN Imputation| G[StandardScaler]
    G --> H([Model Inference])

    style A fill:#fafafa,stroke:#333,stroke-width:2px,color:#000
    style C fill:#ede7f6,stroke:#4527a0,stroke-width:3px,color:#000
    style F fill:#e3f2fd,stroke:#1565c0,stroke-width:3px,color:#000
    style H fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
```

---

## 4. Model Benchmarking Comparison
*A visual representation of how your models perform compared to the baseline. (You can also easily recreate this as a bar chart in Excel/Word).*

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'arial'}}}%%
xychart-beta
    title "Fog Recall vs Mean Absolute Error"
    x-axis ["XGBoost Baseline", "V3.1 (Champion)", "V5 (High Recall)", "Dynamic Hybrid"]
    y-axis "MAE (Meters)" 100 --> 200
    bar [180.5, 127.5, 141.6, 127.2]
    line [2.8, 27.1, 32.8, 29.8]
```
*(Note: The bar represents MAE (lower is better), and the line represents Fog Recall percentage (higher is better)).*

---

## 5. Required Application Screenshots (To take manually)

Since your project includes a Flutter frontend, I highly recommend capturing the following actual screenshots from your running app to put into Chapter 2:
1. **The Main Map View (Light Mode)**: Showing the OpenStreetMap topography and the 10 runway ICAO markers.
2. **The Main Map View (Dark Mode)**: Demonstrating the theme-aware premium aesthetics (CartoDB Dark).
3. **The Horizon Slider**: An action shot showing the bottom control panel (transparent) with the slider shifted to "+3h".
4. **The Auto-Recenter Feature**: A screenshot showing the map panned away with the auto-recenter target icon visible.
5. **The Application Logo**: The `flutter_app/assets/RVR_logo.png` file, which you can use in the introduction or cover page of your report.

## 6. Training & Validation Curves (To add manually)
**Yes!** Including the training and testing curves is highly recommended for Chapter 2.4 (Testing) or Chapter 2.6. They visually prove to the evaluators that your LSTM models successfully converged and did not overfit on the 2024 validation and 2025 test datasets. You should export those plots from your Jupyter notebooks or PyTorch training scripts and add them to the report.
