# AgriSens Core — Technical Research & Methodological Overview

## 1. Core Problem & Objective
In semi-arid and agricultural regions, detecting crop failure early is critical for food security and water management. Traditional ground observations are slow and spatially limited. Optical satellite sensors can track vegetation greenness, but greenness alone is a lagging indicator—plants often retain structural leaf color for days after severe heat or moisture stress has already begun damaging yield.

AgriSens Core solves this by pairing optical vegetation indices with local temperature $z$-scores and precipitation metrics. By analyzing atmospheric thermal anomalies alongside multi-spectral bands, the pipeline flags crop canopy stress before visual damage appears on ground inspection.

---

## 2. Mathematical Framework & Index Calculations

All computations run dynamically on tabular geospatial and satellite feeds (`satellite_crop_data.csv`). The core pipeline executes three primary statistical and band-math operations:

### A. Normalized Difference Vegetation Index (NDVI)
NDVI measures plant photosynthetic activity by comparing Near-Infrared (NIR) light reflection against Red light absorption by chlorophyll:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

* **Input Data:** `NIR` and `Red` spectral reflectance values.
* **Interpretation:** Values above $0.5$ indicate dense, healthy canopy. Rapid drops below regional baselines signal leaf wilting, senescence, or physical destruction.

### B. Thermal Anomaly Z-Score ($Z_{\text{temp}}$)
To identify localized heat shocks independent of normal seasonal variation, Land Surface Temperature ($\text{LST}$) is normalized against rolling 30-day regional baselines (`temp_mean_30d` and `temp_std_30d`):

$$Z_{\text{temp}} = \frac{\text{LST}_{\text{observed}} - \text{LST}_{\text{mean}}}{\text{LST}_{\text{std}}}$$

* **Calculation:** Expresses how many standard deviations the current surface temperature deviates from the 30-day rolling mean.
* **Interpretation:** $Z_{\text{temp}} > +1.5\sigma$ triggers early thermal stress warnings, indicating stomatal throttling where plants restrict water loss and cease photosynthesis.

### C. Integrated Crop Stress Index (ICSI)
To prevent reliance on any single metric, the system synthesizes spectral greenness, heat deviation, and 30-day accumulated rainfall (`precip_mm`) into a continuous composite stress score:

$$\text{ICSI} = w_1 \cdot (1 - \text{NDVI}_{\text{norm}}) + w_2 \cdot \text{Normalize}(Z_{\text{temp}}) + w_3 \cdot \text{Normalize}(\text{Precip}_{\text{deficit}})$$

Where weights $w_1, w_2, w_3$ balance optical response, heat spikes, and moisture scarcity to produce a unified risk tier (Normal, Watch, Warning, Critical).

---

## 3. Data Pipeline & System Execution

1. **Ingestion & Preprocessing:** Loads tabular satellite and weather records using Pandas, filtering missing or corrupt band observations.
2. **Feature Engineering:** Vectorized NumPy operations calculate pixel/coordinate-level NDVI, temperature $z$-scores, and rainfall deficits across the temporal dataset.
3. **Risk Categorization & Visualization:** Outputs processed DataFrames to `data/processed/`, driving interactive charts in Matplotlib and Streamlit dashboard maps.

---

## 4. Git Execution
Once `RESEARCH.md` is updated, execute:
1. `git add RESEARCH.md`
2. `git commit -m "docs: align RESEARCH.md directly with python data pipeline code"`
3. `git push origin main`

Print the terminal output confirming the push to GitHub.
