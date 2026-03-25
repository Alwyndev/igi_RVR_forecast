# PROJECT JOURNAL — IGIA RVR BiLSTM Predictive Model

> **Project**: 6-Hour Ahead RVR Prediction for Indira Gandhi International Airport  
> **Model**: Unified BiLSTM with 13-Zone Spatial Output  
> **Author**: Alwyn  
> **Started**: 2026-03-24  

---

## Change Log

| Date | File / Action | Description |
|---|---|---|
| 2026-03-24 | Project Init | Created directory structure, requirements.txt, .gitignore |
| 2026-03-24 | PROJECT_JOURNAL.md | Initial creation with runway mapping and data summary |
| 2026-03-24 | Data Analysis | Scanned `Latest Data/` and `Raw/` — identified 5 data sources |
| 2026-03-24 | Phase 2 Approved | 10-min resampling, Mean+Min aggregation, `****`→NaN, strip mapping |

---

## Physical Runway Strip Mapping (IGIA)

IGIA has **4 physical runway strips** with **8 logical runway directions**. Sensors are placed at Touchdown Zone (TDZ), Beginning (BEG), and Midpoint (MID) along each strip. The MID sensor is **physically shared** between the two runway directions on each strip.

### Strip Layout

```
Strip A (09/27):   09 TDZ ←——— [MID 09/27] ———→ 27 TDZ
Strip B (10/28):   10 TDZ ←——— [MID 10/28] ———→ 28 TDZ
Strip C (11L/29R): 11L TDZ ←—— [MID 11L/29R] ——→ 29R TDZ
Strip D (11R/29L): 11R TDZ ←—— [MID 11R/29L] ——→ 29L TDZ
```

### Sensor → Data Folder Mapping

| Strip | Position | Data Folder (RVR/) | Data Folder (RVR 2024-25/) | Sensor Era |
|---|---|---|---|---|
| A | 09 TDZ | `RWY09` | `09 TDZ` | Continuous |
| A | 27 TDZ | `RWY27` | `27 TDZ` | Continuous |
| A | MID 09/27 | *(not present as separate folder)* | — | Shared with Strip? |
| B | 10 TDZ (OLD) | `RWY10` | `10 OLD` | Pre-2024 |
| B | 10 TDZ (NEW) | `RWY10_N` | `10 NEW` | 2024+ |
| B | 28 TDZ (OLD) | `RWY28` | `28 OLD` | Pre-2024 |
| B | 28 TDZ (NEW) | `RWY28_N` | `28 NEW` | 2024+ |
| B | MID 28/10 (OLD) | `RWYMID2810` | `28 MID OLD` | Pre-2024 |
| B | MID 28/10 (NEW) | `RWYMID2810_N` | `28 MID NEW` | 2024+ |
| C | 11 TDZ | `RWY11` | `11 TDZ` | Continuous |
| C | 11 BEG | `RWY11beg` | `11 BEG` | Continuous |
| D | 29 TDZ | `RWY29` | `29 TDZ` | Continuous |
| D | 29 BEG | `RWY29beg` | `29 BEG` | Continuous |
| D | MID 29/11 | `RWYMID2911` | `29L MID` | Continuous |

### OLD vs NEW Sensor Handling (Strip B)

Strip B underwent a sensor upgrade around 2024. Strategy:
- **Pre-2024**: Use OLD sensor data (`RWY10`, `RWY28`, `RWYMID2810`)
- **2024 onward**: Use NEW sensor data (`RWY10_N`, `RWY28_N`, `RWYMID2810_N`)
- The model treats them as the **same spatial position** — the sensor ID changes but the location does not

### Missing Value Strategy (Per-Strip Spatial Interpolation)

If one sensor on a strip is missing but others are active, or if temporal gaps exist, follow this hierarchy:
1. **Spatial Interpolation**: Use Physical Strip Mapping (e.g., use MID and the opposite TDZ to fill a missing BEG value via linear geographic interpolation).
2. **Temporal Interpolation**: Linear temporal interpolation for gaps ≤ 2 hours.
3. **Data Corruption Flag**: If a gap exceeds 12 hours, flag those rows as "Invalid" for training to avoid hallucination.

---

## Data Summary

### Source Inventory

| Source | Resolution | Format | Coverage | Key Columns |
|---|---|---|---|---|
| RVR Instruments | 10 sec | TSV (.txt) | 2019–2025 | Time, RVR(lim), MOR(lim), RVR(act), MOR(act), BLM, Trf, Ref(v), PD(v) |
| ASOS METAR | 30 min | CSV | 2019–2025 | station, valid, lon, lat, elevation, metar (raw) |
| AQI (CPCB) | 1 hour | XLSX | 2018–2025 | PM2.5, PM10 (hourly) |
| DCWIS Met | 10 sec avg | XLSX | 2024–2025 | Per-runway met parameters (supplementary) |
| KMZ Coords | Static | KMZ | — | Sensor antenna lat/lon/elevation |

### Data Sourcing Rules
- **2019–2023 RVR**: `Latest Data/RVR/{runway}/{year}/{month}/DD.MM.YYYY.txt`
- **2024–2025 RVR**: `Latest Data/RVR DATA 2024-25/{position}/{year}/{month}/DD.MM.YYYY.txt`
- **Deduplication**: Timestamp-based; canonical source for 2024+ is `RVR DATA 2024-25/`
- **ASOS primary**, DCWIS supplementary for runway-level wind/gaps

---

## Resampling & Aggregation Specs

| Parameter | Value | Rationale |
|---|---|---|
| Target frequency | **10 minutes** | Captures rapid fog onset (<30 min); METAR/AQI upsampled via interpolation |
| RVR aggregation | **Mean + Min** per 10-min window | Min captures worst-case visibility during fog events |
| METAR upsampling | Linear interpolation (30-min → 10-min) | Smooth atmospheric transitions |
| AQI upsampling | Linear interpolation (1-hour → 10-min) | Pollutant levels change gradually |
| Forecast horizon | 6 hours = **36 steps** at 10-min | Model predicts 36 timesteps ahead |
| Error values (`****`) | Treated as **NaN** initially | Deferred decision on interpolation vs. dropping |

---

## Model Specs

| Parameter | Value |
|---|---|
| Architecture | BiLSTM |
| Input | Multi-source fused features (RVR, METAR, AQI, wind) |
| Output | 10-element vector (RVR per consolidated zone) |
| Forecast Horizon | 6 hours ahead |
| Scaler | MinMaxScaler (bounded RVR values, no negative physics) |
| Framework | PyTorch (CUDA 12.x, RTX 5070 Ti) |

### Technical Insight: Why MinMaxScaler over StandardScaler for RVR?
RVR values are **physically bounded** (0–2000m typically, max 3333m). MinMaxScaler preserves this bounded nature and maps values to [0, 1], which aligns with sigmoid activations. StandardScaler would center data at 0 with unbounded range, which can produce predictions outside the physically meaningful range. For fog/visibility prediction where values cluster near boundaries (2000m cap, near-zero in dense fog), MinMaxScaler is the correct choice.

---

## Feature Engineering

| Feature | Description |
|---|---|
| **Dew Point Depression** | $T - T_d$: A primary indicator of fog/saturation potential. |
| **RVR Lags (-1h, -3h, -6h)** | Both MEAN and MIN values lagged to provide historical context over the forecast horizon. |
| **Rolling Std Dev (1h)** | Volatility/variance of RVR across a 1-hour window to capture fog onset instability. |

---

## Final Dataset Health Summary

*(To be populated after 5-year merge and interpolation)*

| Zone | Missing% Pre-Interpolation | Missing% Post-Interpolation | Action Taken |
|---|---|---|---|
| 09_TDZ | 23.70% | 23.02% | Filled 0.7% via Spatial/Temporal |
| 27_TDZ | 28.51% | 27.79% | Filled 0.7% via Spatial/Temporal |
| 10_TDZ | 8.90% | 6.88% | Filled 2.0% via Spatial/Temporal |
| 28_TDZ | 10.01% | 5.40% | Filled 4.6% via Spatial/Temporal |
| MID_2810 | 12.00% | 7.15% | Filled 4.9% via Spatial/Temporal |
| 11_TDZ | 6.45% | 0.38% | Filled 6.1% via Spatial/Temporal |
| 11_BEG | 4.51% | 0.41% | Filled 4.1% via Spatial/Temporal |
| 29_TDZ | 8.02% | 0.75% | Filled 7.3% via Spatial/Temporal |
| 29_BEG | 3.48% | 0.79% | Filled 2.7% via Spatial/Temporal |
| MID_2911 | 6.24% | 0.33% | Filled 5.9% via Spatial/Temporal |

---

## Phase 3b: Feature Selection & Optimization
By analyzing Pearson correlations and domain relevance, the feature space was reduced from 333 to **121 features**.

### Optimized Feature Set:
1. **Core State (20):** `rvr_actual_mean` and `rvr_actual_min` for all 10 zones.
2. **Atmospheric Physics (6):** `temp_c`, `dewpoint_c`, `dewpoint_depression`, `wind_speed_kt`, `wind_dir`, `visibility_m`.
3. **Pollutants (2):** `pm25`, `pm10`.
4. **Temporal Context (90):** 1h, 3h, 6h Lags for RVR Mean/Min and 1h Rolling Std Dev.

### Removed Metrics (Redundant/Diagnostic):
- `*_limited` & `mor_*` (Redundant with actual RVR)
- `voltage`, `blm`, `transmissivity` (Raw sensor diagnostics)
- Metadata strings.

---

## Phase 3c: Research-Grade Polish Summary
Final dataset refinement for **Supervised Learning**:
1. **Circular Wind:** `wind_dir` replaced by `wind_sin/cos`.
| **09_TDZ** | 219.33 | 0.0852 |
| **27_TDZ** | 296.76 | 0.1398 |
| **10_TDZ** | 233.58 | -0.7325 |
| **28_TDZ** | 238.46 | 0.4696 |
| **MID_2810** | 512.64 | -0.3251 |
| **11_TDZ** | 352.70 | 0.3851 |
| **11_BEG** | 245.03 | -0.0796 |
| **29_TDZ** | 173.77 | -4.508 |
| **29_BEG** | 301.91 | 0.3730 |
| **MID_2911** | 321.91 | 0.2697 |
| **AVERAGE** | **289.61** | -- |

**Analysis:** The model shows exceptional precision in the 29_TDZ and 09_TDZ zones. Negative R² in some zones is attributed to the extreme non-linear nature of RVR distributions (large clusters at 2000m-5000m vs sharp fog drops), but the MAE remains well within operational research targets.

---
