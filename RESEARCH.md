# AgriSens Core — Technical Research & Methodological Overview

## Abstract
AgriSens Core is an open‑source analytics pipeline that fuses satellite‑derived spectral indices with surface‑temperature and precipitation anomaly metrics to produce an early‑warning stress indicator for agricultural lands. By operating on per‑pixel time series, the system quantifies canopy loss, heat stress, and moisture deficit in a unified composite score, enabling rapid identification of high‑risk zones without requiring on‑ground measurements.

## 1. Problem Statement
In semi‑arid and rain‑fed cropping systems, drought and heat spikes can reduce yields long before visual symptoms appear. Ground‑based observations are sparse and delayed, while raw satellite imagery offers only indirect clues. The research challenge is to translate raw reflectance and thermal observations into a physically‑meaningful, quantitative stress metric that can be computed automatically for any region.

## 2. Data & Methodology
The pipeline ingests a CSV (`satellite_crop_data.csv`) containing calibrated reflectance bands (NIR, Red, Blue), daily land‑surface temperature (LST) in °C, and daily precipitation in mm. Missing values are filtered out during preprocessing.

### 2.1 Normalized Difference Vegetation Index (NDVI)
The NDVI is computed per the classic definition:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

Implemented in `src/analytics/indices.py` using NumPy with explicit zero‑division protection and clipping to the range \([-1,\,1]\).

### 2.2 Land‑Surface Temperature Z‑Score ($Z_{\text{LST}}$)
A rolling 30‑day baseline of LST is maintained for each pixel. The Z‑score quantifies deviation from the local climatology:

$$Z_{\text{LST}} = \frac{\text{LST}_{\text{obs}} - \mu_{\text{LST,30d}}}{\sigma_{\text{LST,30d}}}$$

Values $>+1.5$ indicate statistically significant heat stress.

### 2.3 Soil‑Moisture Z‑Score ($Z_{\text{SM}}$)
Analogous to the temperature score, soil‑moisture (derived from the `GWETTOP` field of the NASA POWER API) is normalised:

$$Z_{\text{SM}} = \frac{\text{SM}_{\text{obs}} - \mu_{\text{SM,30d}}}{\sigma_{\text{SM,30d}}}$$

Negative values signal moisture deficit.

### 2.4 Integrated Crop Stress Index (ICSI) / Composite Agricultural Risk Index (CARI)
The core composite metric blends three risk components using a logistic sigmoid transform $\Phi(z) = \frac{1}{1+e^{-z}}$ (implemented in `RiskEngine.logistic_transform`). The final score is scaled to \([0,100]\):

$$\begin{aligned}
\Delta_{\text{NDVI}} &= \frac{\text{NDVI}_{\text{obs}} - \text{NDVI}_{\text{baseline}}}{\text{NDVI}_{\text{baseline}}} \\
\text{CanopyRisk} &= \Phi\bigl(-5\,\Delta_{\text{NDVI}}\bigr) \\
\text{MoistureRisk} &= \Phi\bigl(-Z_{\text{SM}}\bigr) \\
\text{HeatRisk} &= \Phi\bigl(Z_{\text{LST}}\bigr) \\
\text{CARI} &= \bigl[0.40\times\text{CanopyRisk} + 0.35\times\text{MoistureRisk} + 0.25\times\text{HeatRisk}\bigr]\times100
\end{aligned}$$

The implementation lives in `src/analytics/risk_engine.py`. The score is rounded to two decimals before classification.

## 3. Risk Indicator Matrix
| Component | Input Variable | Transformation | Weight | Thresholds (Risk Tier) |
|-----------|----------------|----------------|--------|-----------------------|
| Canopy Loss | $\Delta_{\text{NDVI}}$ | $\Phi(-5\Delta_{\text{NDVI}})$ | 0.40 | $\Delta_{\text{NDVI}} < -0.2$ yields high canopy risk |
| Soil‑Moisture Deficit | $Z_{\text{SM}}$ | $\Phi(-Z_{\text{SM}})$ | 0.35 | $Z_{\text{SM}} < -1.5$ ⇒ strong deficit |
| Heat Stress | $Z_{\text{LST}}$ | $\Phi(Z_{\text{LST}})$ | 0.25 | $Z_{\text{LST}} > +1.5$ ⇒ severe heat |

The overall CARI tiers are:
- **NOMINAL**: CARI < 30
- **MODERATE STRESS**: 30 ≤ CARI < 55
- **HIGH RISK**: 55 ≤ CARI < 75
- **CRITICAL RISK**: CARI ≥ 75

## 4. Limitations
- **Temporal Resolution**: Daily satellite composites limit detection of sub‑daily heat spikes.
- **Cloud Contamination**: NDVI is masked only for obvious cloud pixels; thin clouds can still bias reflectance.
- **Single‑Sensor Dependence**: Current implementation relies on MODIS‑style band set; other sensors require additional band‑order handling.

## 5. Future Extensions
- **Synthetic Aperture Radar (SAR) Integration** – SAR backscatter can provide moisture information under cloud cover.
- **Dynamic Phenology Curves** – Incorporate vegetation phenology models to adjust baseline NDVI per growth stage.
- **Real‑time Sensor Streams** – Fuse IoT soil‑moisture probes for higher‑frequency moisture anomalies.
- **Multi‑spectral Fusion** – Extend the index suite with red‑edge and SWIR bands for improved stress discrimination.

---
*All formulas correspond line‑for‑line to the Python implementation in the repository.*
